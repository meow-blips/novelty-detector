"""
novelty_detector/ingestion/preprocessor.py
============================================
Ingestion & Preprocessing Module (Phase 1)
-------------------------------------------
Responsibilities:
  1. Accept raw text strings from any source (file, API, DB row, etc.).
  2. Perform Unicode normalisation and basic character-level cleaning.
  3. Tokenise with NLTK (fast, dependency-free fallback) or spaCy.
  4. Remove stopwords and apply stemming/lemmatisation.
  5. Return a cleaned token list and a normalised string ready for shingling.

Design Decisions
----------------
- The `TextPreprocessor` class is intentionally stateless between documents
  so it is safe to use in a multi-threaded Flask context.
- All heavy NLTK resources are loaded once at class instantiation, not per
  call, to minimise latency in hot-path processing.
- spaCy is an *optional* upgrade path; the class gracefully falls back to
  NLTK if the spaCy model is not installed.
- Raises `PreprocessingError` (a project-specific exception) on unrecoverable
  failures so callers can catch a single, predictable exception type.

Example
-------
    from novelty_detector.ingestion.preprocessor import TextPreprocessor

    prep = TextPreprocessor()
    result = prep.process("The quick brown fox jumps over the lazy dog!")
    print(result.tokens)      # ['quick', 'brown', 'fox', 'jump', 'lazi', 'dog']
    print(result.clean_text)  # "quick brown fox jump lazi dog"
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

# ── NLTK imports ──────────────────────────────────────────────────────────────
# We import lazily inside __init__ to give a clear error if NLTK data is missing.
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

# ── loguru for structured logging ─────────────────────────────────────────────
from loguru import logger

# ── Project-level config ──────────────────────────────────────────────────────
from novelty_detector.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class PreprocessingError(Exception):
    """Raised when a document cannot be preprocessed due to an unrecoverable error."""


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessedDocument:
    """
    Container returned by ``TextPreprocessor.process()``.

    Attributes
    ----------
    doc_id : str
        The identifier supplied by the caller (e.g., filename, UUID).
    raw_text : str
        The original, unmodified input text.
    clean_text : str
        A single normalised string of space-joined stemmed tokens.
        This is what gets shingled downstream.
    tokens : List[str]
        Individual stemmed / lemmatised tokens (no stopwords).
    token_count : int
        Number of tokens *after* cleaning; used to reject stub documents.
    """
    doc_id: str
    raw_text: str
    clean_text: str
    tokens: List[str]
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = len(self.tokens)

    def is_valid(self, min_tokens: int = 5) -> bool:
        """Return True if the document has enough tokens to be meaningful."""
        return self.token_count >= min_tokens


# ─────────────────────────────────────────────────────────────────────────────
# Main Preprocessor class
# ─────────────────────────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    Stateless text preprocessing pipeline.

    Parameters
    ----------
    language : str
        ISO 639-1 language code for NLTK stopwords (default: ``"english"``).
    use_stemming : bool
        If True (default) apply Porter stemming. Set to False to keep
        base-form tokens (e.g., for transformer-based downstream steps that
        prefer natural language).
    use_spacy : bool
        If True, attempt to load a spaCy model for lemmatisation instead of
        NLTK stemming. Falls back to NLTK on ImportError / OSError.
    spacy_model : str
        spaCy model name (default: ``"en_core_web_sm"``). Only used when
        ``use_spacy=True``.
    extra_stopwords : Optional[List[str]]
        Domain-specific stopwords to add on top of NLTK's list.

    Raises
    ------
    PreprocessingError
        If neither NLTK nor spaCy is available, or if required NLTK corpora
        are missing.
    """

    # Regex to keep only ASCII letters, digits, apostrophes, and whitespace.
    # Apostrophes are retained to handle contractions before tokenisation.
    _KEEP_CHARS = re.compile(r"[^a-z0-9'\s]")

    # Collapse runs of whitespace to a single space.
    _WHITESPACE = re.compile(r"\s+")

    def __init__(
        self,
        language: str = "english",
        use_stemming: bool = True,
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm",
        extra_stopwords: Optional[List[str]] = None,
    ) -> None:

        if not _NLTK_AVAILABLE:
            raise PreprocessingError(
                "NLTK is not installed. Run: pip install nltk"
            )

        self.language = language
        self.use_stemming = use_stemming

        # ── Load NLTK resources ───────────────────────────────────────────────
        self._ensure_nltk_resources()

        # Build the stopword set (hash set for O(1) lookup).
        self._stop_words: set[str] = set(stopwords.words(language))
        if extra_stopwords:
            self._stop_words.update(w.lower() for w in extra_stopwords)

        # ── Stemmer ───────────────────────────────────────────────────────────
        self._stemmer: Optional[PorterStemmer] = (
            PorterStemmer() if use_stemming else None
        )

        # ── Optional spaCy backend ────────────────────────────────────────────
        self._nlp = None  # spaCy Language object or None
        if use_spacy:
            self._nlp = self._load_spacy(spacy_model)

        logger.info(
            "TextPreprocessor initialised | language={} stemming={} spaCy={}",
            language,
            use_stemming,
            self._nlp is not None,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, raw_text: str, doc_id: str = "unknown") -> PreprocessedDocument:
        """
        Full preprocessing pipeline for a single document.

        Steps
        -----
        1. Unicode normalise (NFKD → ASCII fold).
        2. Lower-case.
        3. Strip non-alphanumeric characters.
        4. Tokenise (spaCy or NLTK).
        5. Remove stopwords.
        6. Stem / lemmatise.
        7. Build clean_text string.

        Parameters
        ----------
        raw_text : str
            The raw document content.
        doc_id : str
            A caller-supplied identifier (filename, UUID, row ID, …).

        Returns
        -------
        PreprocessedDocument
            A validated result object ready for shingling.

        Raises
        ------
        PreprocessingError
            If ``raw_text`` is not a string, or if any pipeline step fails
            in an unexpected way.
        """
        if not isinstance(raw_text, str):
            raise PreprocessingError(
                f"raw_text must be a str, got {type(raw_text).__name__!r} "
                f"(doc_id={doc_id!r})"
            )

        if not raw_text.strip():
            raise PreprocessingError(
                f"raw_text is empty or whitespace-only (doc_id={doc_id!r})"
            )

        try:
            # Step 1 & 2: Unicode normalisation + lower-case
            normalised = self._unicode_normalise(raw_text)

            # Step 3: Strip non-alphanumeric characters
            cleaned = self._strip_noise(normalised)

            # Step 4: Tokenise
            if self._nlp is not None:
                tokens_raw = self._spacy_tokenise(cleaned)
            else:
                tokens_raw = self._nltk_tokenise(cleaned)

            # Step 5: Remove stopwords (single-char tokens also removed here)
            tokens_no_stop = [
                t for t in tokens_raw
                if t not in self._stop_words and len(t) > 1
            ]

            # Step 6: Stem or lemmatise
            final_tokens = self._normalise_tokens(tokens_no_stop)

            # Step 7: Reconstruct clean string
            clean_text = " ".join(final_tokens)

            doc = PreprocessedDocument(
                doc_id=doc_id,
                raw_text=raw_text,
                clean_text=clean_text,
                tokens=final_tokens,
            )

            if not doc.is_valid(settings.min_token_count):
                logger.warning(
                    "Document '{}' has only {} token(s) after cleaning "
                    "(threshold={}). It may produce unreliable similarity scores.",
                    doc_id,
                    doc.token_count,
                    settings.min_token_count,
                )

            logger.debug(
                "Preprocessed doc_id={!r} | raw_chars={} → tokens={}",
                doc_id,
                len(raw_text),
                doc.token_count,
            )
            return doc

        except PreprocessingError:
            raise  # re-raise project errors unchanged
        except Exception as exc:
            raise PreprocessingError(
                f"Unexpected error preprocessing doc_id={doc_id!r}: {exc}"
            ) from exc

    def process_batch(
        self, documents: List[tuple[str, str]]
    ) -> List[PreprocessedDocument]:
        """
        Convenience wrapper to process multiple documents.

        Parameters
        ----------
        documents : List[tuple[str, str]]
            Each element is ``(doc_id, raw_text)``.

        Returns
        -------
        List[PreprocessedDocument]
            Successfully processed documents. Failed documents are logged
            and skipped (not raised) to keep batch jobs resilient.
        """
        results: List[PreprocessedDocument] = []
        for doc_id, raw_text in documents:
            try:
                results.append(self.process(raw_text, doc_id=doc_id))
            except PreprocessingError as exc:
                logger.error("Skipping doc_id={!r}: {}", doc_id, exc)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_nltk_resources() -> None:
        """
        Download required NLTK data packages if they are not already present.
        This is idempotent — NLTK's downloader skips already-cached data.
        """
        required = [
            ("tokenizers/punkt_tab", "punkt_tab"),
            ("corpora/stopwords", "stopwords"),
        ]
        for resource_path, download_id in required:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                logger.info("Downloading NLTK resource: {}", download_id)
                nltk.download(download_id, quiet=True)

    @staticmethod
    def _unicode_normalise(text: str) -> str:
        """
        NFKD normalise then ASCII-encode to strip accented characters.

        Examples
        --------
        "Ré­sumé" → "Resume"
        "naïve"   → "naive"
        """
        # NFKD decomposes characters into base + combining marks.
        normalised = unicodedata.normalize("NFKD", text)
        # Encode to ASCII (ignoring non-ASCII bytes) then decode back.
        ascii_text = normalised.encode("ascii", errors="ignore").decode("ascii")
        return ascii_text.lower()

    def _strip_noise(self, text: str) -> str:
        """
        Remove characters that are not letters, digits, or apostrophes.
        Then collapse whitespace.
        """
        text = self._KEEP_CHARS.sub(" ", text)
        text = self._WHITESPACE.sub(" ", text)
        return text.strip()

    @staticmethod
    def _nltk_tokenise(text: str) -> List[str]:
        """Word-tokenise using NLTK's Punkt tokeniser."""
        return word_tokenize(text)

    def _spacy_tokenise(self, text: str) -> List[str]:
        """
        Lemmatise using spaCy. Returns the lemma for content words
        and falls back to the lower-cased surface form for others.
        """
        doc = self._nlp(text)
        return [
            token.lemma_ if token.lemma_ != "-PRON-" else token.lower_
            for token in doc
            if not token.is_space
        ]

    def _normalise_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming (Porter) or return tokens as-is if disabled."""
        if self._stemmer is not None:
            return [self._stemmer.stem(t) for t in tokens]
        return tokens

    @staticmethod
    def _load_spacy(model_name: str):
        """
        Attempt to load a spaCy Language model.

        Returns
        -------
        spacy.Language or None
            None if spaCy is not installed or the model is missing.
        """
        try:
            import spacy  # noqa: PLC0415 — intentional lazy import
            nlp = spacy.load(model_name)
            logger.info("spaCy model '{}' loaded successfully.", model_name)
            return nlp
        except ImportError:
            logger.warning("spaCy not installed; falling back to NLTK tokeniser.")
        except OSError:
            logger.warning(
                "spaCy model '{}' not found. "
                "Run: python -m spacy download {}",
                model_name,
                model_name,
            )
        return None
