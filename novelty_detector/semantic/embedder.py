"""
novelty_detector/semantic/embedder.py
=======================================
Semantic Verification Module — Stage 2
---------------------------------------
After the LSH lexical filter (Stage 1) returns a small candidate set,
this module re-scores each candidate pair using dense sentence embeddings
and cosine similarity to catch **paraphrased near-duplicates** that share
few surface-level n-grams but carry the same meaning.

How it works
------------
1. Load a pre-trained ``sentence-transformers`` model once at startup.
2. For every (query_doc, candidate_doc) pair flagged by LSH:
   a. Encode both documents' *clean_text* into fixed-dimension embedding vectors.
   b. Compute the **cosine similarity** between the two vectors (range −1 → 1;
      for sentence embeddings in practice 0 → 1).
3. Return a ``SemanticScore`` result that the Classifier will threshold.

Model choice
------------
``all-MiniLM-L6-v2`` is the default:
  - 22 M parameters — fits comfortably in CPU RAM.
  - 384-dimensional embeddings.
  - Strong general-purpose semantic performance (MTEB benchmark).
  - ~5× faster than large models on CPU.

Override via ``EMBEDDING_MODEL`` in ``.env`` (any HuggingFace SentenceTransformer).

Batching
--------
``encode_batch`` is preferred over repeated ``encode`` calls.
``sentence-transformers`` handles internal batching automatically; we
expose ``batch_size`` to let callers tune throughput vs. latency.

Example
-------
    from novelty_detector.semantic.embedder import SemanticEmbedder

    emb = SemanticEmbedder()
    score = emb.score_pair(clean_text_a, clean_text_b)
    print(score.cosine_similarity)   # e.g. 0.923
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from novelty_detector.config import settings

# ── sentence-transformers is an optional dependency at import time ─────────────
try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class EmbedderError(Exception):
    """Raised when the embedding or similarity computation fails."""


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SemanticScore:
    """
    Holds the cosine similarity result for one document pair.

    Attributes
    ----------
    query_doc_id : str
        Identifier of the query document.
    candidate_doc_id : str
        Identifier of the candidate document from the index.
    cosine_similarity : float
        Cosine similarity in [−1, 1].  For sentence embeddings this is
        effectively in [0, 1]; values >= ``settings.semantic_threshold``
        indicate a near-duplicate.
    embedding_dim : int
        Dimensionality of the embedding vectors (diagnostic).
    """

    query_doc_id: str
    candidate_doc_id: str
    cosine_similarity: float
    embedding_dim: int

    @property
    def is_near_duplicate(self) -> bool:
        """True if cosine similarity meets the configured threshold."""
        return self.cosine_similarity >= settings.semantic_threshold

    def __repr__(self) -> str:
        return (
            f"SemanticScore("
            f"query={self.query_doc_id!r}, "
            f"candidate={self.candidate_doc_id!r}, "
            f"cosine={self.cosine_similarity:.4f}, "
            f"near_dup={self.is_near_duplicate})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity helpers (pure NumPy — no SciPy required)
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Numerically stable: avoids division by zero via a small epsilon guard.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        1-D float arrays of the same shape.

    Returns
    -------
    float
        Cosine similarity in [−1, 1].
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        # One or both vectors are zero-length (degenerate embedding).
        logger.warning(
            "Near-zero embedding norm encountered (norm_a={:.2e}, norm_b={:.2e}). "
            "Returning similarity=0.",
            norm_a, norm_b,
        )
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _pairwise_cosine_matrix(
    embeddings_a: np.ndarray, embeddings_b: np.ndarray
) -> np.ndarray:
    """
    Compute an (m × n) cosine similarity matrix between two embedding arrays.

    Parameters
    ----------
    embeddings_a : np.ndarray
        Shape (m, d).
    embeddings_b : np.ndarray
        Shape (n, d).

    Returns
    -------
    np.ndarray
        Shape (m, n) with values in [−1, 1].
    """
    # L2-normalise each row so that dot product == cosine similarity.
    norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True).clip(min=1e-10)
    norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True).clip(min=1e-10)
    normed_a = embeddings_a / norms_a
    normed_b = embeddings_b / norms_b
    return normed_a @ normed_b.T  # (m, d) @ (d, n) = (m, n)


# ─────────────────────────────────────────────────────────────────────────────
# Main Embedder class
# ─────────────────────────────────────────────────────────────────────────────

class SemanticEmbedder:
    """
    Thin wrapper around ``sentence-transformers`` for semantic similarity scoring.

    Parameters
    ----------
    model_name : str
        HuggingFace SentenceTransformer model identifier.
        Default: ``settings.embedding_model`` (``"all-MiniLM-L6-v2"``).
    device : str or None
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        If None, automatically selects CUDA if available, else CPU.
    cache_embeddings : bool
        If True, previously computed embeddings are stored in an in-process
        LRU cache keyed by doc_id.  Useful when the same documents are
        queried repeatedly (e.g., in the Flask dashboard).  Default: True.

    Raises
    ------
    EmbedderError
        If ``sentence-transformers`` is not installed, or the model cannot
        be loaded from the HuggingFace hub.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        device: Optional[str] = None,
        cache_embeddings: bool = True,
    ) -> None:

        if not _ST_AVAILABLE:
            raise EmbedderError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.cache_embeddings = cache_embeddings

        # In-process embedding cache: doc_id → np.ndarray
        self._cache: dict[str, np.ndarray] = {} if cache_embeddings else {}

        # ── Resolve device ────────────────────────────────────────────────────
        if device is None:
            try:
                import torch  # noqa: PLC0415
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.device = device

        # ── Load model ────────────────────────────────────────────────────────
        logger.info(
            "Loading SentenceTransformer model '{}' on device='{}'…",
            model_name, device,
        )
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbedderError(
                f"Failed to load model '{model_name}': {exc}"
            ) from exc

        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded | dim={} | device={}",
            self.embedding_dim, device,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        doc_id: Optional[str] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Encode a single text string into a dense embedding vector.

        Parameters
        ----------
        text : str
            The *clean_text* from a ``PreprocessedDocument`` (or raw text).
        doc_id : str or None
            Optional cache key.  If provided and ``cache_embeddings=True``,
            the result is stored so subsequent calls with the same ``doc_id``
            return instantly.
        batch_size : int
            Passed through to the underlying model (unused for single texts
            but kept for API consistency).

        Returns
        -------
        np.ndarray
            1-D float32 array of shape ``(embedding_dim,)``.

        Raises
        ------
        EmbedderError
            If ``text`` is empty or encoding fails.
        """
        if not text or not text.strip():
            raise EmbedderError(
                f"Cannot encode empty text (doc_id={doc_id!r})."
            )

        # Cache hit?
        if self.cache_embeddings and doc_id and doc_id in self._cache:
            logger.debug("Cache hit for doc_id={!r}", doc_id)
            return self._cache[doc_id]

        try:
            # encode() returns a list or np.ndarray depending on input type.
            # Passing a plain string → shape (dim,) array.
            vec: np.ndarray = self._model.encode(
                text,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,  # we normalise ourselves for control
            )
        except Exception as exc:
            raise EmbedderError(
                f"Encoding failed for doc_id={doc_id!r}: {exc}"
            ) from exc

        if self.cache_embeddings and doc_id:
            self._cache[doc_id] = vec

        logger.debug(
            "Encoded doc_id={!r} | dim={} | norm={:.4f}",
            doc_id, vec.shape[0], float(np.linalg.norm(vec)),
        )
        return vec

    def encode_batch(
        self,
        texts_with_ids: Sequence[Tuple[str, str]],
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        """
        Encode multiple texts in one forward pass (efficient batching).

        Parameters
        ----------
        texts_with_ids : Sequence[Tuple[str, str]]
            Each element is ``(doc_id, clean_text)``.
        batch_size : int
            Number of texts per model forward pass.

        Returns
        -------
        dict[str, np.ndarray]
            Maps ``doc_id`` → embedding vector.  Entries that fail are logged
            and excluded (not raised) to keep batch jobs resilient.
        """
        # Separate cached from uncached.
        to_encode: List[Tuple[str, str]] = []
        results: dict[str, np.ndarray] = {}

        for doc_id, text in texts_with_ids:
            if self.cache_embeddings and doc_id in self._cache:
                results[doc_id] = self._cache[doc_id]
            elif text.strip():
                to_encode.append((doc_id, text))
            else:
                logger.warning(
                    "Skipping empty text for doc_id={!r} in batch encode.", doc_id
                )

        if to_encode:
            ids, texts = zip(*to_encode)
            try:
                vecs: np.ndarray = self._model.encode(
                    list(texts),
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=len(to_encode) > 20,
                    normalize_embeddings=False,
                )
                for doc_id, vec in zip(ids, vecs):
                    results[doc_id] = vec
                    if self.cache_embeddings:
                        self._cache[doc_id] = vec
            except Exception as exc:
                logger.error("Batch encoding failed: {}", exc)

        logger.info(
            "Encoded {}/{} texts (batch_size={}).",
            len(results), len(texts_with_ids), batch_size,
        )
        return results

    def score_pair(
        self,
        text_a: str,
        text_b: str,
        doc_id_a: Optional[str] = None,
        doc_id_b: Optional[str] = None,
    ) -> SemanticScore:
        """
        Compute cosine similarity between two texts.

        Parameters
        ----------
        text_a : str
            Clean text of the query document.
        text_b : str
            Clean text of the candidate document.
        doc_id_a, doc_id_b : str or None
            Optional identifiers (used for logging and caching).

        Returns
        -------
        SemanticScore
            Contains the cosine similarity and a convenience ``is_near_duplicate``
            property based on ``settings.semantic_threshold``.

        Raises
        ------
        EmbedderError
            If either text is empty or encoding fails.
        """
        vec_a = self.encode(text_a, doc_id=doc_id_a)
        vec_b = self.encode(text_b, doc_id=doc_id_b)
        similarity = _cosine_similarity(vec_a, vec_b)

        score = SemanticScore(
            query_doc_id=doc_id_a or "unknown_a",
            candidate_doc_id=doc_id_b or "unknown_b",
            cosine_similarity=similarity,
            embedding_dim=self.embedding_dim,
        )

        logger.debug(
            "Semantic score {} ↔ {}: cosine={:.4f} (threshold={})",
            doc_id_a, doc_id_b, similarity, settings.semantic_threshold,
        )
        return score

    def score_candidates(
        self,
        query_text: str,
        candidate_texts: Sequence[Tuple[str, str]],
        query_doc_id: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[SemanticScore]:
        """
        Score a query document against multiple candidates in a single batch
        encode — the efficient path for Stage 2 verification.

        Parameters
        ----------
        query_text : str
            Clean text of the query document.
        candidate_texts : Sequence[Tuple[str, str]]
            Each element is ``(candidate_doc_id, candidate_clean_text)``.
        query_doc_id : str or None
            Optional identifier for the query document.
        batch_size : int
            Batch size for the model forward pass.

        Returns
        -------
        List[SemanticScore]
            One ``SemanticScore`` per candidate, sorted by descending
            cosine similarity.

        Raises
        ------
        EmbedderError
            If the query text is empty or encoding fails.
        """
        if not candidate_texts:
            return []

        # Encode query
        query_vec = self.encode(query_text, doc_id=query_doc_id, batch_size=batch_size)

        # Batch-encode all candidates
        cand_ids = [cid for cid, _ in candidate_texts]
        cand_texts = [ct for _, ct in candidate_texts]

        try:
            cand_vecs: np.ndarray = self._model.encode(
                cand_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
        except Exception as exc:
            raise EmbedderError(
                f"Batch candidate encoding failed (query={query_doc_id!r}): {exc}"
            ) from exc

        # Compute cosine similarities (query vs each candidate)
        # query_vec shape: (dim,)  →  expand to (1, dim) for matrix multiply
        sim_row = _pairwise_cosine_matrix(
            query_vec.reshape(1, -1), cand_vecs
        )[0]  # shape: (n_candidates,)

        scores = [
            SemanticScore(
                query_doc_id=query_doc_id or "unknown",
                candidate_doc_id=cid,
                cosine_similarity=float(sim),
                embedding_dim=self.embedding_dim,
            )
            for cid, sim in zip(cand_ids, sim_row)
        ]

        # Sort descending by cosine similarity
        scores.sort(key=lambda s: s.cosine_similarity, reverse=True)

        logger.info(
            "Scored {} candidate(s) for query={!r} | "
            "top cosine={:.4f}",
            len(scores),
            query_doc_id,
            scores[0].cosine_similarity if scores else 0.0,
        )
        return scores

    def clear_cache(self) -> None:
        """Evict all cached embeddings (e.g., to free memory between batches)."""
        self._cache.clear()
        logger.debug("Embedding cache cleared.")
