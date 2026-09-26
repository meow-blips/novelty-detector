"""
novelty_detector/pipeline.py
==============================
End-to-End Pipeline Orchestrator
----------------------------------
Wires together all four Phase 1 & 2 modules into a single ``Pipeline``
class that is the **primary public API** for the whole system.

Callers (CLI scripts, Flask routes, batch jobs) need only interact with
``Pipeline`` — they don't need to know about the individual modules.

Two-Phase Flow
--------------

    ┌─────────────────────────────────────────────────────────────┐
    │  Raw text + doc_id                                          │
    │       │                                                     │
    │  [1] TextPreprocessor.process()                             │
    │       │   → PreprocessedDocument                            │
    │       │                                                     │
    │  [2] LSHIndex.query()                         Stage 1       │
    │       │   → List[CandidateMatch]  (may be [])               │
    │       │                                                     │
    │  [3] SemanticEmbedder.score_candidates()      Stage 2       │
    │       │   → List[SemanticScore]               (LSH > 0)     │
    │       │                                                     │
    │  [4] DocumentClassifier.process_document()                  │
    │       │   → ClassificationResult  (+ DB write)              │
    │       │                                                     │
    │  [5] LSHIndex.add_document()                                │
    │           (only if document is novel or operator allows dup) │
    └─────────────────────────────────────────────────────────────┘

Usage
-----
    from novelty_detector.pipeline import Pipeline

    pipeline = Pipeline()

    # Index some existing corpus documents first
    pipeline.index_corpus([
        ("doc_1", "The cat sat on the mat."),
        ("doc_2", "Quantum entanglement in modern physics."),
    ])

    # Process a new incoming document
    result = pipeline.process("A cat was sitting on a mat.", doc_id="new_doc")
    print(result.verdict)        # 'near-duplicate'
    print(result.best_match_id)  # 'doc_1'
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from novelty_detector.classification.classifier import (
    ClassificationResult,
    DocumentClassifier,
)
from novelty_detector.config import settings
from novelty_detector.ingestion.preprocessor import PreprocessedDocument, TextPreprocessor
from novelty_detector.lexical.lsh_index import LSHIndex
from novelty_detector.semantic.embedder import SemanticEmbedder


class PipelineError(Exception):
    """Raised when the pipeline encounters an unrecoverable error."""


class Pipeline:
    """
    Full two-phase novelty and duplicate detection pipeline.

    Parameters
    ----------
    preprocessor : TextPreprocessor or None
        Supply a pre-configured instance, or leave None to use defaults.
    lsh_index : LSHIndex or None
        Supply a pre-built / loaded index, or leave None to create fresh.
    embedder : SemanticEmbedder or None
        Supply a pre-loaded embedder, or leave None to load the default model.
        Pass a dummy object with a ``score_candidates`` method to skip Stage 2.
    classifier : DocumentClassifier or None
        Supply a pre-configured classifier, or leave None to use defaults.
    index_novel_documents : bool
        If True (default), documents classified as 'novel' are automatically
        added to the LSH index so future documents can match against them.
    index_near_duplicates : bool
        If True, near-duplicates are also added to the index (riskier —
        can cause index bloat if many paraphrases enter the corpus).
        Default: False.
    skip_semantic : bool
        If True, skip Stage 2 entirely (faster, less accurate).
        Useful for very large corpora where semantic scoring is too slow.

    Thread-safety
    -------------
    ``LSHIndex`` is NOT thread-safe.  Wrap the ``process`` call in a lock
    when using Pipeline under concurrent Flask workers, or switch to a
    Redis-backed LSH index.
    """

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        lsh_index: Optional[LSHIndex] = None,
        embedder: Optional[SemanticEmbedder] = None,
        classifier: Optional[DocumentClassifier] = None,
        index_novel_documents: bool = True,
        index_near_duplicates: bool = False,
        skip_semantic: bool = False,
    ) -> None:

        logger.info("Initialising Pipeline…")

        self.preprocessor = preprocessor or TextPreprocessor()
        self.lsh_index = lsh_index or LSHIndex()
        self.classifier = classifier or DocumentClassifier()
        self.index_novel_documents = index_novel_documents
        self.index_near_duplicates = index_near_duplicates
        self.skip_semantic = skip_semantic

        # Embedder loads the model — defer if skipping semantic stage.
        if not skip_semantic:
            self.embedder = embedder or SemanticEmbedder()
        else:
            self.embedder = None
            logger.warning(
                "Semantic stage is DISABLED (skip_semantic=True). "
                "Only Jaccard similarity will be used for classification."
            )

        logger.info("Pipeline ready. Index size: {}", self.lsh_index.size)

    # ─────────────────────────────────────────────────────────────────────────
    # Corpus Indexing
    # ─────────────────────────────────────────────────────────────────────────

    def index_corpus(
        self,
        documents: List[Tuple[str, str]],
        skip_errors: bool = True,
    ) -> int:
        """
        Preprocess and index a batch of existing corpus documents.

        These are documents already known to the system (e.g., a seeded
        corpus). They are indexed in the LSH structure but **not** run
        through the classifier (no verdicts, no DB writes).

        Parameters
        ----------
        documents : List[Tuple[str, str]]
            Each element is ``(doc_id, raw_text)``.
        skip_errors : bool
            If True, log failures and continue; if False, raise on first error.

        Returns
        -------
        int
            Number of documents successfully indexed.
        """
        logger.info("Indexing corpus of {} document(s)…", len(documents))

        preprocessed = self.preprocessor.process_batch(documents)
        results = self.lsh_index.add_documents_batch(preprocessed)
        success_count = sum(1 for v in results.values() if v is not None)

        logger.info(
            "Corpus indexing complete: {}/{} documents indexed.",
            success_count, len(documents),
        )
        return success_count

    # ─────────────────────────────────────────────────────────────────────────
    # Single Document Processing
    # ─────────────────────────────────────────────────────────────────────────

    def process(
        self,
        raw_text: str,
        doc_id: str = "unknown",
    ) -> ClassificationResult:
        """
        Run a single document through the full two-phase pipeline.

        Parameters
        ----------
        raw_text : str
            The raw input text.
        doc_id : str
            A stable, unique identifier for this document.

        Returns
        -------
        ClassificationResult
            The verdict, scores, per-pair breakdowns, and DB-persisted record.

        Raises
        ------
        PipelineError
            If a non-recoverable error occurs at any stage.
        """
        logger.info("─── Processing doc_id={!r} ───", doc_id)

        # ── Stage 0: Preprocess ───────────────────────────────────────────────
        try:
            doc: PreprocessedDocument = self.preprocessor.process(
                raw_text, doc_id=doc_id
            )
        except Exception as exc:
            raise PipelineError(
                f"Preprocessing failed for doc_id={doc_id!r}: {exc}"
            ) from exc

        # ── Stage 1: LSH Lexical Filter ───────────────────────────────────────
        try:
            lsh_candidates = self.lsh_index.query(doc, exclude_self=True)
        except Exception as exc:
            raise PipelineError(
                f"LSH query failed for doc_id={doc_id!r}: {exc}"
            ) from exc

        logger.info(
            "Stage 1 complete: {} LSH candidate(s) for doc_id={!r}",
            len(lsh_candidates), doc_id,
        )

        # ── Stage 2: Semantic Verification ────────────────────────────────────
        semantic_scores = []
        if lsh_candidates and not self.skip_semantic and self.embedder is not None:
            # Only score candidates that did NOT already clear the Jaccard bar
            # (those would be classified as 'duplicate' regardless).
            to_score = [
                c for c in lsh_candidates
                if c.estimated_jaccard < settings.duplicate_jaccard_threshold
            ]
            if to_score:
                try:
                    # Fetch clean_text for each candidate from the classifier's
                    # storage layer (or ask caller to supply a corpus map).
                    # For now we rely on the embedder's cache / re-encode.
                    candidate_texts = self._fetch_candidate_texts(to_score)

                    semantic_scores = self.embedder.score_candidates(
                        query_text=doc.clean_text,
                        candidate_texts=candidate_texts,
                        query_doc_id=doc_id,
                    )
                except Exception as exc:
                    logger.error(
                        "Stage 2 semantic scoring failed for doc_id={!r}: {}. "
                        "Falling back to Stage 1 classification only.",
                        doc_id, exc,
                    )

        logger.info(
            "Stage 2 complete: {} semantic score(s) for doc_id={!r}",
            len(semantic_scores), doc_id,
        )

        # ── Classification ────────────────────────────────────────────────────
        result: ClassificationResult = self.classifier.process_document(
            doc=doc,
            lsh_candidates=lsh_candidates,
            semantic_scores=semantic_scores,
        )

        # ── Selective Re-Indexing ─────────────────────────────────────────────
        # Add the document to the LSH index based on its verdict,
        # unless it's already there (e.g., re-submitted corpus doc).
        if not self.lsh_index.contains(doc_id):
            should_index = (
                (result.verdict == "novel" and self.index_novel_documents)
                or (result.verdict == "near-duplicate" and self.index_near_duplicates)
            )
            if should_index:
                try:
                    self.lsh_index.add_document(doc)
                    logger.debug(
                        "doc_id={!r} added to LSH index (verdict={}).",
                        doc_id, result.verdict,
                    )
                except Exception as exc:
                    logger.error(
                        "Could not add doc_id={!r} to LSH index: {}", doc_id, exc
                    )

        return result

    def process_batch(
        self, documents: List[Tuple[str, str]]
    ) -> List[ClassificationResult]:
        """
        Process multiple documents in sequence.

        Parameters
        ----------
        documents : List[Tuple[str, str]]
            Each element is ``(doc_id, raw_text)``.

        Returns
        -------
        List[ClassificationResult]
            Results for each document that was successfully processed.
            Failed documents are logged and skipped.
        """
        results: List[ClassificationResult] = []
        for i, (doc_id, raw_text) in enumerate(documents, start=1):
            logger.info("Batch progress: {}/{}", i, len(documents))
            try:
                results.append(self.process(raw_text, doc_id=doc_id))
            except PipelineError as exc:
                logger.error("Skipping doc_id={!r}: {}", doc_id, exc)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_candidate_texts(
        self, candidates: list
    ) -> List[Tuple[str, str]]:
        """
        Retrieve clean_text for each candidate document from the DB.

        Falls back to the doc_id itself if the DB has no record (possible
        during corpus bootstrapping before the first DB write).
        """
        from novelty_detector.storage.database import get_session
        from novelty_detector.storage.models import DocumentRecord

        candidate_ids = [c.candidate_doc_id for c in candidates]
        texts: dict[str, str] = {}

        try:
            with get_session() as session:
                records = (
                    session.query(DocumentRecord)
                    .filter(DocumentRecord.doc_id.in_(candidate_ids))
                    .all()
                )
                for r in records:
                    if r.clean_text:
                        texts[r.doc_id] = r.clean_text
        except Exception as exc:
            logger.warning(
                "Could not fetch candidate texts from DB: {}. "
                "Semantic stage may be partial.", exc
            )

        # For any candidates not found in the DB, use the doc_id as a placeholder.
        result_pairs: List[Tuple[str, str]] = []
        for cand_id in candidate_ids:
            text = texts.get(cand_id, "")
            if text:
                result_pairs.append((cand_id, text))
            else:
                logger.warning(
                    "No clean_text found for candidate doc_id={!r}; "
                    "skipping from semantic scoring.",
                    cand_id,
                )
        return result_pairs
