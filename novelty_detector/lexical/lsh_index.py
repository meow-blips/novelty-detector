"""
novelty_detector/lexical/lsh_index.py
=======================================
Lexical Filter — MinHash + LSH Index (Phase 1)
-----------------------------------------------
This module is the core of **Stage 1** in the duplicate detection pipeline.

How it works
------------
1. A caller passes a ``PreprocessedDocument`` (from the ingestion module).
2. The document's tokens are shingled (overlapping n-grams).
3. A **MinHash** signature is computed from the shingle set using ``num_perm``
   independent hash functions.  MinHash is an unbiased estimator of the
   *Jaccard similarity* between two sets:

       J(A, B) = |A ∩ B| / |A ∪ B|

4. The MinHash is inserted into a **Locality-Sensitive Hashing (LSH)** index.
   LSH partitions the MinHash signature into *bands*; two documents collide
   into the same bucket if at least one band is identical.  This gives
   sub-linear query time O(1) instead of O(n) pairwise comparisons.

5. On query, the index returns the ``doc_id``s of *candidate* near-duplicates.
   These candidates are then passed to Stage 2 (semantic verification).

Key design choices
------------------
- **datasketch** is the industry-standard Python MinHash/LSH library and is
  used here for correctness and performance.
- The index is held **in-memory** for Phase 1.  Phase 2 will add a Redis or
  SQLite-backed persistent store.
- ``LSHIndex`` is intentionally *not* thread-safe by default.  Wrap it in a
  lock or use ``datasketch.MinHashLSH`` with ``storage_config`` for concurrent
  Flask workers.
- Each indexed document also stores its raw ``MinHash`` object so that the
  *exact* estimated Jaccard score can be computed at query time (useful for
  the classification thresholds).

Example
-------
    from novelty_detector.ingestion.preprocessor import TextPreprocessor
    from novelty_detector.lexical.lsh_index import LSHIndex

    prep = TextPreprocessor()
    idx  = LSHIndex()

    doc_a = prep.process("The cat sat on the mat.", doc_id="doc_a")
    doc_b = prep.process("A cat was sitting on a mat.", doc_id="doc_b")
    doc_c = prep.process("Quantum entanglement is a physics concept.", doc_id="doc_c")

    idx.add_document(doc_a)
    idx.add_document(doc_b)
    idx.add_document(doc_c)

    candidates = idx.query(doc_b)
    # → [CandidateMatch(doc_id='doc_a', estimated_jaccard=0.63, ...)]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from datasketch import MinHash, MinHashLSH
from loguru import logger

from novelty_detector.config import settings
from novelty_detector.ingestion.preprocessor import PreprocessedDocument, PreprocessingError
from novelty_detector.lexical.shingler import ShinglingError, build_shingles


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class LSHIndexError(Exception):
    """Raised when the LSH index encounters an unrecoverable error."""


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(order=True)
class CandidateMatch:
    """
    Represents one candidate near-duplicate pair returned by the LSH query.

    Attributes
    ----------
    query_doc_id : str
        The doc_id of the document being queried.
    candidate_doc_id : str
        The doc_id of the candidate already in the index.
    estimated_jaccard : float
        MinHash-estimated Jaccard similarity (0–1).
        Note: this is a *probabilistic estimate*, not the true Jaccard.
        The true Jaccard would require comparing the full shingle sets.
    shingle_count_query : int
        Number of unique shingles in the query document (diagnostic).
    """
    # Sort by descending similarity for convenient display.
    estimated_jaccard: float
    query_doc_id: str = field(compare=False)
    candidate_doc_id: str = field(compare=False)
    shingle_count_query: int = field(compare=False)

    def __repr__(self) -> str:
        return (
            f"CandidateMatch("
            f"query={self.query_doc_id!r}, "
            f"candidate={self.candidate_doc_id!r}, "
            f"jaccard≈{self.estimated_jaccard:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MinHash builder helper
# ─────────────────────────────────────────────────────────────────────────────

def build_minhash(shingles: Set[str], num_perm: int) -> MinHash:
    """
    Construct a ``datasketch.MinHash`` signature from a shingle set.

    Parameters
    ----------
    shingles : Set[str]
        The set of n-gram shingles (output of ``build_shingles``).
    num_perm : int
        Number of hash permutations.  Must be >= 2.  128 is a good default:
        it gives ~1% Jaccard estimation error at reasonable memory cost.

    Returns
    -------
    MinHash
        A signed MinHash object ready to be inserted into the LSH index.

    Raises
    ------
    LSHIndexError
        If ``shingles`` is empty or ``num_perm < 2``.

    Notes
    -----
    Each shingle is encoded to UTF-8 bytes before being hashed, so the
    signature is deterministic across Python processes.
    """
    if num_perm < 2:
        raise LSHIndexError(f"num_perm must be >= 2, got {num_perm}.")

    if not shingles:
        raise LSHIndexError(
            "Cannot build a MinHash from an empty shingle set. "
            "Ensure the document has enough tokens."
        )

    minhash = MinHash(num_perm=num_perm)
    for shingle in shingles:
        # Encode each shingle to bytes (required by datasketch).
        minhash.update(shingle.encode("utf-8"))

    return minhash


# ─────────────────────────────────────────────────────────────────────────────
# LSH Index
# ─────────────────────────────────────────────────────────────────────────────

class LSHIndex:
    """
    In-memory MinHash + LSH index for fast candidate retrieval.

    Parameters
    ----------
    threshold : float
        Jaccard similarity threshold for LSH band partitioning.
        Documents above this threshold are considered candidates.
        Default: ``settings.lsh_threshold`` (0.5).
    num_perm : int
        Number of MinHash hash functions.
        Default: ``settings.minhash_num_perm`` (128).
    shingle_size : int
        Size of n-gram shingles.
        Default: ``settings.shingle_size`` (3).

    Thread-safety
    -------------
    Not thread-safe by default.  Use ``threading.Lock`` around ``add_document``
    and ``query`` when running under multi-threaded Flask workers, or switch
    to a Redis-backed ``MinHashLSH`` storage backend.

    Persistence
    -----------
    Phase 1 uses in-memory storage.  Serialise with ``pickle`` if you need
    to persist between process restarts (see ``save`` / ``load`` class methods).
    """

    def __init__(
        self,
        threshold: float = settings.lsh_threshold,
        num_perm: int = settings.minhash_num_perm,
        shingle_size: int = settings.shingle_size,
    ) -> None:

        # Validate parameters early to surface config mistakes at startup.
        if not (0.0 < threshold < 1.0):
            raise LSHIndexError(
                f"LSH threshold must be in (0, 1), got {threshold}."
            )
        if num_perm < 2:
            raise LSHIndexError(f"num_perm must be >= 2, got {num_perm}.")
        if shingle_size < 1:
            raise LSHIndexError(f"shingle_size must be >= 1, got {shingle_size}.")

        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size

        # Core datasketch LSH structure.
        # The threshold drives automatic band/row partitioning:
        #   b * r = num_perm,  threshold ≈ (1/b)^(1/r)
        self._lsh: MinHashLSH = MinHashLSH(
            threshold=threshold,
            num_perm=num_perm,
        )

        # Side-store: map doc_id → MinHash so we can compute exact pairwise
        # Jaccard estimates at query time (not just bucket membership).
        self._signatures: Dict[str, MinHash] = {}

        # Track all indexed doc_ids for membership checks.
        self._indexed_ids: Set[str] = set()

        logger.info(
            "LSHIndex created | threshold={} num_perm={} shingle_size={}",
            threshold,
            num_perm,
            shingle_size,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Return the number of documents currently in the index."""
        return len(self._indexed_ids)

    def contains(self, doc_id: str) -> bool:
        """Return True if ``doc_id`` is already in the index."""
        return doc_id in self._indexed_ids

    def add_document(self, doc: PreprocessedDocument) -> MinHash:
        """
        Shingle, fingerprint, and index a pre-processed document.

        Parameters
        ----------
        doc : PreprocessedDocument
            Output of ``TextPreprocessor.process()``.

        Returns
        -------
        MinHash
            The MinHash signature that was inserted (useful for callers that
            want to cache or inspect the signature).

        Raises
        ------
        LSHIndexError
            - If ``doc.doc_id`` is already indexed (duplicates not allowed).
            - If the document is too short to produce any shingles.
            - If any lower-level shingling or MinHash step fails.

        Notes
        -----
        This method is intentionally *not* idempotent.  Call ``remove_document``
        first if you need to re-index a document with updated content.
        """
        doc_id = doc.doc_id

        # Guard: prevent double-indexing the same doc_id.
        if self.contains(doc_id):
            raise LSHIndexError(
                f"doc_id {doc_id!r} is already in the index. "
                "Call remove_document() before re-indexing."
            )

        # ── Step 1: Shingle ───────────────────────────────────────────────────
        try:
            shingles = build_shingles(doc.tokens, n=self.shingle_size)
        except ShinglingError as exc:
            raise LSHIndexError(
                f"Failed to shingle doc_id={doc_id!r}: {exc}"
            ) from exc

        # ── Step 2: Build MinHash ─────────────────────────────────────────────
        try:
            minhash = build_minhash(shingles, num_perm=self.num_perm)
        except LSHIndexError:
            raise  # already a project error, pass through

        # ── Step 3: Insert into LSH index ─────────────────────────────────────
        try:
            self._lsh.insert(doc_id, minhash)
        except Exception as exc:
            raise LSHIndexError(
                f"datasketch LSH insertion failed for doc_id={doc_id!r}: {exc}"
            ) from exc

        # ── Step 4: Store the signature for pairwise Jaccard lookups ──────────
        self._signatures[doc_id] = minhash
        self._indexed_ids.add(doc_id)

        logger.debug(
            "Indexed doc_id={!r} | shingles={} | index_size={}",
            doc_id,
            len(shingles),
            self.size,
        )
        return minhash

    def add_documents_batch(
        self, docs: List[PreprocessedDocument]
    ) -> Dict[str, Optional[MinHash]]:
        """
        Index a list of documents, logging but not raising on individual failures.

        Parameters
        ----------
        docs : List[PreprocessedDocument]
            List of preprocessed documents.

        Returns
        -------
        Dict[str, Optional[MinHash]]
            Maps ``doc_id`` → ``MinHash`` on success, or ``None`` on failure.
        """
        results: Dict[str, Optional[MinHash]] = {}
        for doc in docs:
            try:
                results[doc.doc_id] = self.add_document(doc)
            except LSHIndexError as exc:
                logger.error("Failed to index doc_id={!r}: {}", doc.doc_id, exc)
                results[doc.doc_id] = None
        logger.info(
            "Batch indexing complete: {}/{} documents indexed successfully.",
            sum(1 for v in results.values() if v is not None),
            len(docs),
        )
        return results

    def query(
        self,
        doc: PreprocessedDocument,
        exclude_self: bool = True,
    ) -> List[CandidateMatch]:
        """
        Find candidate near-duplicates for a document.

        The document does **not** need to be in the index — this supports both
        lookup-before-insert and post-hoc queries.

        Parameters
        ----------
        doc : PreprocessedDocument
            The query document (from the preprocessor).
        exclude_self : bool
            If True (default), exclude ``doc.doc_id`` from results if it is
            already in the index (avoids self-match).

        Returns
        -------
        List[CandidateMatch]
            Candidate matches sorted by descending estimated Jaccard similarity.
            An empty list means the document is *novel* at the lexical level.

        Raises
        ------
        LSHIndexError
            If shingling or MinHash construction fails for the query document.

        Notes
        -----
        The index must have at least one document before querying.
        """
        if self.size == 0:
            logger.warning("LSH index is empty — no candidates possible.")
            return []

        doc_id = doc.doc_id

        # ── Build query MinHash (same pipeline as indexing) ───────────────────
        try:
            shingles = build_shingles(doc.tokens, n=self.shingle_size)
        except ShinglingError as exc:
            raise LSHIndexError(
                f"Failed to shingle query doc_id={doc_id!r}: {exc}"
            ) from exc

        try:
            query_minhash = build_minhash(shingles, num_perm=self.num_perm)
        except LSHIndexError:
            raise

        # ── Query the LSH index for bucket collisions ─────────────────────────
        try:
            raw_candidates: List[str] = self._lsh.query(query_minhash)
        except Exception as exc:
            raise LSHIndexError(
                f"datasketch LSH query failed for doc_id={doc_id!r}: {exc}"
            ) from exc

        # ── Score each candidate ──────────────────────────────────────────────
        matches: List[CandidateMatch] = []
        for cand_id in raw_candidates:
            # Optionally skip self-matches.
            if exclude_self and cand_id == doc_id:
                continue

            # Retrieve the stored MinHash for this candidate.
            cand_minhash = self._signatures.get(cand_id)
            if cand_minhash is None:
                # Should never happen; defensive guard.
                logger.error(
                    "Candidate doc_id={!r} is in LSH but has no stored signature. "
                    "Index may be corrupted.",
                    cand_id,
                )
                continue

            # Compute estimated Jaccard similarity using MinHash.
            # This is O(num_perm) — very fast.
            estimated_jaccard: float = query_minhash.jaccard(cand_minhash)

            matches.append(
                CandidateMatch(
                    query_doc_id=doc_id,
                    candidate_doc_id=cand_id,
                    estimated_jaccard=estimated_jaccard,
                    shingle_count_query=len(shingles),
                )
            )

        # Sort by descending similarity (highest match first).
        matches.sort(reverse=True)

        logger.info(
            "Query doc_id={!r} → {} candidate(s) from index of {} docs.",
            doc_id,
            len(matches),
            self.size,
        )
        return matches

    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document from the index.

        Parameters
        ----------
        doc_id : str
            The identifier of the document to remove.

        Raises
        ------
        LSHIndexError
            If ``doc_id`` is not in the index.
        """
        if not self.contains(doc_id):
            raise LSHIndexError(
                f"Cannot remove doc_id={doc_id!r}: not found in index."
            )

        minhash = self._signatures.pop(doc_id)
        self._indexed_ids.discard(doc_id)

        try:
            self._lsh.remove(doc_id)
        except Exception as exc:
            # Roll back our side-store changes to keep state consistent.
            self._signatures[doc_id] = minhash
            self._indexed_ids.add(doc_id)
            raise LSHIndexError(
                f"datasketch LSH removal failed for doc_id={doc_id!r}: {exc}"
            ) from exc

        logger.debug("Removed doc_id={!r} from index.", doc_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence helpers (Phase 1 — in-process pickle)
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Serialise the entire index to a pickle file.

        Parameters
        ----------
        path : str
            File path to write to (e.g., ``"data/lsh_index.pkl"``).

        Warnings
        --------
        Pickle files are not secure against untrusted data. Replace with a
        Redis-backed store or a proper serialisation format in production.
        """
        import pickle  # noqa: PLC0415

        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "threshold": self.threshold,
                    "num_perm": self.num_perm,
                    "shingle_size": self.shingle_size,
                    "lsh": self._lsh,
                    "signatures": self._signatures,
                    "indexed_ids": self._indexed_ids,
                },
                fh,
            )
        logger.info("LSHIndex saved to '{}' ({} documents).", path, self.size)

    @classmethod
    def load(cls, path: str) -> "LSHIndex":
        """
        Deserialise an index from a pickle file created by ``save()``.

        Parameters
        ----------
        path : str
            Path to the pickle file.

        Returns
        -------
        LSHIndex
            A fully reconstructed index ready for querying.
        """
        import pickle  # noqa: PLC0415

        with open(path, "rb") as fh:
            state = pickle.load(fh)

        instance = cls.__new__(cls)
        instance.threshold = state["threshold"]
        instance.num_perm = state["num_perm"]
        instance.shingle_size = state["shingle_size"]
        instance._lsh = state["lsh"]
        instance._signatures = state["signatures"]
        instance._indexed_ids = state["indexed_ids"]

        logger.info(
            "LSHIndex loaded from '{}' ({} documents).", path, instance.size
        )
        return instance
