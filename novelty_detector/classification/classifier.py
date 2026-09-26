"""
novelty_detector/classification/classifier.py
================================================
Classification & Audit Logging Module (Phase 2)
-------------------------------------------------
This module is the decision-making brain of the pipeline.

Responsibility
--------------
Given the raw scores produced by Stage 1 (Jaccard) and Stage 2 (cosine),
apply configurable thresholds to assign one of three labels to an incoming
document relative to each candidate:

    'duplicate'      — very high Jaccard (near-exact copy, minimal rephrasing)
    'near-duplicate' — high cosine similarity (paraphrase / slight rewrite)
    'novel'          — does not match any indexed document at either threshold

The overall verdict for the *document* is the most severe verdict across all
candidate pairs (duplicate > near-duplicate > novel).

Classification Logic
--------------------

    if jaccard >= DUPLICATE_JACCARD_THRESHOLD:
        verdict = 'duplicate'          # Stage 1 is enough; skip Stage 2

    elif cosine >= SEMANTIC_THRESHOLD:
        verdict = 'near-duplicate'     # Caught by Stage 2

    else:
        verdict = 'novel'              # Both stages passed; no match found

Audit Log
---------
Every pair comparison and final verdict is persisted via the storage layer
(``ComparisonLog`` and ``DocumentRecord`` ORM models).

Flask Integration
-----------------
``DocumentClassifier`` is stateless between calls — safe to instantiate once
in the Flask app factory and share across requests.

Example
-------
    from novelty_detector.classification.classifier import DocumentClassifier

    clf = DocumentClassifier()
    result = clf.process_document(
        doc=preprocessed_doc,
        lsh_candidates=lsh_matches,     # from LSHIndex.query()
        semantic_scores=semantic_scores, # from SemanticEmbedder.score_candidates()
    )
    print(result.verdict)          # 'novel' | 'near-duplicate' | 'duplicate'
    print(result.best_match_id)    # doc_id of the closest match, or None
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from novelty_detector.config import settings
from novelty_detector.ingestion.preprocessor import PreprocessedDocument
from novelty_detector.lexical.lsh_index import CandidateMatch
from novelty_detector.semantic.embedder import SemanticScore
from novelty_detector.storage.database import get_session, init_db
from novelty_detector.storage.models import ComparisonLog, DocumentRecord


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exception
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationError(Exception):
    """Raised when classification cannot be completed."""


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Comprehensive result for one processed document.

    Attributes
    ----------
    doc_id : str
        The document's identifier.
    verdict : str
        Overall document verdict: ``'novel'``, ``'near-duplicate'``, or
        ``'duplicate'``.
    best_match_id : Optional[str]
        doc_id of the highest-scoring candidate, or None if novel.
    best_jaccard : Optional[float]
        Jaccard estimate for the best match (Stage 1 score).
    best_cosine : Optional[float]
        Cosine similarity for the best match (Stage 2 score); None if
        duplicate was decided on Jaccard alone.
    pair_verdicts : List[PairVerdict]
        Detailed verdict for every candidate pair examined.
    jaccard_threshold_used : float
        The ``duplicate_jaccard_threshold`` active at classification time.
    semantic_threshold_used : float
        The ``semantic_threshold`` active at classification time.
    """

    doc_id: str
    verdict: str
    best_match_id: Optional[str]
    best_jaccard: Optional[float]
    best_cosine: Optional[float]
    pair_verdicts: List["PairVerdict"] = field(default_factory=list)
    jaccard_threshold_used: float = field(
        default_factory=lambda: settings.duplicate_jaccard_threshold
    )
    semantic_threshold_used: float = field(
        default_factory=lambda: settings.semantic_threshold
    )

    def __repr__(self) -> str:
        return (
            f"<ClassificationResult doc_id={self.doc_id!r} "
            f"verdict={self.verdict!r} "
            f"best_match={self.best_match_id!r} "
            f"jaccard={self.best_jaccard} "
            f"cosine={self.best_cosine}>"
        )


@dataclass
class PairVerdict:
    """
    The verdict for a single (query, candidate) pair.

    Attributes
    ----------
    query_doc_id        The incoming document.
    candidate_doc_id    The matched corpus document.
    jaccard_estimate    MinHash Jaccard from Stage 1.
    cosine_similarity   Cosine similarity from Stage 2 (None if not needed).
    verdict             'duplicate' | 'near-duplicate' | 'novel'
    stage_decided       Which stage made the final call: 1 or 2.
    """

    query_doc_id: str
    candidate_doc_id: str
    jaccard_estimate: float
    cosine_similarity: Optional[float]
    verdict: str
    stage_decided: int  # 1 = LSH decided, 2 = Semantic decided


# ─────────────────────────────────────────────────────────────────────────────
# Severity ordering (for choosing the overall document verdict)
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY: dict[str, int] = {
    "novel": 0,
    "near-duplicate": 1,
    "duplicate": 2,
}


def _most_severe(verdicts: List[str]) -> str:
    """Return the most severe verdict from a list. Defaults to 'novel'."""
    if not verdicts:
        return "novel"
    return max(verdicts, key=lambda v: _SEVERITY.get(v, 0))


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

class DocumentClassifier:
    """
    Stateless classifier that combines LSH and semantic scores into a verdict,
    persists the results to the database, and returns a structured result.

    Parameters
    ----------
    duplicate_jaccard_threshold : float
        Jaccard ≥ this → ``'duplicate'`` (Stage 1 decides, Stage 2 skipped).
        Default: ``settings.duplicate_jaccard_threshold`` (0.9).
    semantic_threshold : float
        Cosine ≥ this → ``'near-duplicate'``.
        Default: ``settings.semantic_threshold`` (0.85).
    persist : bool
        If True (default), write ``DocumentRecord`` and ``ComparisonLog``
        entries to the database.  Set False for unit tests or dry-runs.

    Thread-safety
    -------------
    Stateless — safe to share across Flask request threads.
    Each DB write uses ``get_session()`` which creates an independent session.
    """

    def __init__(
        self,
        duplicate_jaccard_threshold: float = settings.duplicate_jaccard_threshold,
        semantic_threshold: float = settings.semantic_threshold,
        persist: bool = True,
    ) -> None:

        self.duplicate_jaccard_threshold = duplicate_jaccard_threshold
        self.semantic_threshold = semantic_threshold
        self.persist = persist

        if persist:
            # Ensure tables exist on first use.
            init_db()

        logger.info(
            "DocumentClassifier ready | "
            "jaccard_dup_threshold={} | semantic_threshold={} | persist={}",
            duplicate_jaccard_threshold,
            semantic_threshold,
            persist,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Core classification logic (pure, no I/O)
    # ─────────────────────────────────────────────────────────────────────────

    def classify_pair(
        self,
        query_doc_id: str,
        candidate_doc_id: str,
        jaccard_estimate: float,
        cosine_similarity: Optional[float],
    ) -> PairVerdict:
        """
        Apply thresholds to a single (query, candidate) pair.

        Parameters
        ----------
        query_doc_id : str
        candidate_doc_id : str
        jaccard_estimate : float
            MinHash Jaccard from Stage 1.
        cosine_similarity : Optional[float]
            Cosine similarity from Stage 2, or None if Stage 2 was not run.

        Returns
        -------
        PairVerdict
        """
        # ── Stage 1 decision (Jaccard) ────────────────────────────────────────
        if jaccard_estimate >= self.duplicate_jaccard_threshold:
            return PairVerdict(
                query_doc_id=query_doc_id,
                candidate_doc_id=candidate_doc_id,
                jaccard_estimate=jaccard_estimate,
                cosine_similarity=cosine_similarity,
                verdict="duplicate",
                stage_decided=1,
            )

        # ── Stage 2 decision (cosine) ─────────────────────────────────────────
        if cosine_similarity is not None and cosine_similarity >= self.semantic_threshold:
            return PairVerdict(
                query_doc_id=query_doc_id,
                candidate_doc_id=candidate_doc_id,
                jaccard_estimate=jaccard_estimate,
                cosine_similarity=cosine_similarity,
                verdict="near-duplicate",
                stage_decided=2,
            )

        # ── Neither threshold met ─────────────────────────────────────────────
        return PairVerdict(
            query_doc_id=query_doc_id,
            candidate_doc_id=candidate_doc_id,
            jaccard_estimate=jaccard_estimate,
            cosine_similarity=cosine_similarity,
            verdict="novel",
            stage_decided=2 if cosine_similarity is not None else 1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Full document processing
    # ─────────────────────────────────────────────────────────────────────────

    def process_document(
        self,
        doc: PreprocessedDocument,
        lsh_candidates: List[CandidateMatch],
        semantic_scores: Optional[List[SemanticScore]] = None,
    ) -> ClassificationResult:
        """
        Classify a document and persist the result.

        Parameters
        ----------
        doc : PreprocessedDocument
            The preprocessed query document.
        lsh_candidates : List[CandidateMatch]
            Output of ``LSHIndex.query()``.  May be empty (→ novel).
        semantic_scores : Optional[List[SemanticScore]]
            Output of ``SemanticEmbedder.score_candidates()``.
            Pass None or [] to skip Stage 2 (useful for debugging Stage 1).

        Returns
        -------
        ClassificationResult
            Complete classification with per-pair breakdowns.

        Raises
        ------
        ClassificationError
            If the document cannot be classified due to an unrecoverable error.
        """
        doc_id = doc.doc_id

        # ── Build a cosine lookup map: candidate_doc_id → SemanticScore ───────
        cosine_map: dict[str, float] = {}
        if semantic_scores:
            for s in semantic_scores:
                cosine_map[s.candidate_doc_id] = s.cosine_similarity

        # ── Classify each LSH candidate pair ──────────────────────────────────
        pair_verdicts: List[PairVerdict] = []

        for candidate in lsh_candidates:
            cand_id = candidate.candidate_doc_id
            jaccard = candidate.estimated_jaccard
            cosine = cosine_map.get(cand_id)  # None if Stage 2 not run

            pv = self.classify_pair(
                query_doc_id=doc_id,
                candidate_doc_id=cand_id,
                jaccard_estimate=jaccard,
                cosine_similarity=cosine,
            )
            pair_verdicts.append(pv)

            logger.debug(
                "Pair ({!r} ↔ {!r}) | jaccard={:.3f} cosine={} → {}",
                doc_id, cand_id, jaccard,
                f"{cosine:.3f}" if cosine is not None else "N/A",
                pv.verdict,
            )

        # ── Roll up to a document-level verdict ───────────────────────────────
        if not pair_verdicts:
            overall_verdict = "novel"
            best_match_id = None
            best_jaccard = None
            best_cosine = None
        else:
            overall_verdict = _most_severe([pv.verdict for pv in pair_verdicts])

            # Identify the best-matching pair (highest cosine; fall back to Jaccard)
            best_pv = max(
                pair_verdicts,
                key=lambda pv: (
                    pv.cosine_similarity if pv.cosine_similarity is not None else 0.0,
                    pv.jaccard_estimate,
                ),
            )
            best_match_id = best_pv.candidate_doc_id
            best_jaccard = best_pv.jaccard_estimate
            best_cosine = best_pv.cosine_similarity

        result = ClassificationResult(
            doc_id=doc_id,
            verdict=overall_verdict,
            best_match_id=best_match_id,
            best_jaccard=best_jaccard,
            best_cosine=best_cosine,
            pair_verdicts=pair_verdicts,
            jaccard_threshold_used=self.duplicate_jaccard_threshold,
            semantic_threshold_used=self.semantic_threshold,
        )

        logger.info(
            "Classified doc_id={!r} → {} | best_match={!r} | "
            "jaccard={} | cosine={}",
            doc_id,
            overall_verdict,
            best_match_id,
            f"{best_jaccard:.3f}" if best_jaccard is not None else "N/A",
            f"{best_cosine:.3f}" if best_cosine is not None else "N/A",
        )

        # ── Persist to database ───────────────────────────────────────────────
        if self.persist:
            try:
                self._persist_result(doc, result)
            except Exception as exc:
                # Log but do NOT re-raise: a DB failure must not block the
                # classification result from being returned to the caller.
                logger.error(
                    "Failed to persist result for doc_id={!r}: {}", doc_id, exc
                )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _persist_result(
        self,
        doc: PreprocessedDocument,
        result: ClassificationResult,
    ) -> None:
        """
        Write (or update) the ``DocumentRecord`` and all ``ComparisonLog``
        rows for this classification result inside a single transaction.

        This method is idempotent on the ``DocumentRecord`` (upsert via
        merge-on-doc_id check). ``ComparisonLog`` rows use the unique
        constraint on ``(query_doc_id, candidate_doc_id)`` to avoid
        duplicates.
        """
        with get_session() as session:
            # ── Upsert DocumentRecord ─────────────────────────────────────────
            existing = (
                session.query(DocumentRecord)
                .filter_by(doc_id=doc.doc_id)
                .first()
            )

            if existing is None:
                record = DocumentRecord(
                    doc_id=doc.doc_id,
                    submitted_at=datetime.datetime.utcnow(),
                    raw_text=doc.raw_text,
                    clean_text=doc.clean_text,
                    token_count=doc.token_count,
                    status=result.verdict,
                )
                session.add(record)
                logger.debug("Inserted DocumentRecord for doc_id={!r}", doc.doc_id)
            else:
                # Update status if a more severe verdict was found.
                if _SEVERITY.get(result.verdict, 0) > _SEVERITY.get(existing.status, 0):
                    existing.status = result.verdict
                    logger.debug(
                        "Updated DocumentRecord doc_id={!r} status → {}",
                        doc.doc_id, result.verdict,
                    )

            # Flush so the DocumentRecord PK is available for FK references.
            session.flush()

            # ── Insert ComparisonLog rows ──────────────────────────────────────
            for pv in result.pair_verdicts:
                # Check for existing log (unique constraint guard)
                exists = (
                    session.query(ComparisonLog)
                    .filter_by(
                        query_doc_id=pv.query_doc_id,
                        candidate_doc_id=pv.candidate_doc_id,
                    )
                    .first()
                )
                if exists is not None:
                    logger.debug(
                        "ComparisonLog already exists for pair ({!r}, {!r}); skipping.",
                        pv.query_doc_id, pv.candidate_doc_id,
                    )
                    continue

                log_entry = ComparisonLog(
                    query_doc_id=pv.query_doc_id,
                    candidate_doc_id=pv.candidate_doc_id,
                    compared_at=datetime.datetime.utcnow(),
                    jaccard_estimate=pv.jaccard_estimate,
                    cosine_similarity=pv.cosine_similarity,
                    verdict=pv.verdict,
                    lsh_threshold_used=settings.lsh_threshold,
                    semantic_threshold_used=self.semantic_threshold,
                    review_status="pending",
                )
                session.add(log_entry)

            # Session auto-commits on context manager exit.
