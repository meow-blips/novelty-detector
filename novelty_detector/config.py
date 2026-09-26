"""
novelty_detector/config.py
===========================
Central configuration dataclass.
Values can be overridden via environment variables or a .env file.

Usage
-----
    from novelty_detector.config import Settings
    cfg = Settings()
    print(cfg.lsh_threshold)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()  # reads .env from the project root if present


@dataclass
class Settings:
    # ── Preprocessing ─────────────────────────────────────────────────────────
    # Minimum number of tokens a document must have after cleaning.
    # Documents shorter than this are considered too short to analyse.
    min_token_count: int = int(os.getenv("MIN_TOKEN_COUNT", "5"))

    # ── Lexical Filter (MinHash / LSH) ────────────────────────────────────────
    # Number of hash functions used per MinHash signature.
    # Higher → more accurate Jaccard estimate, but slower & more memory.
    minhash_num_perm: int = int(os.getenv("MINHASH_NUM_PERM", "128"))

    # n-gram size for shingling (2 = bigrams, 3 = trigrams, …).
    shingle_size: int = int(os.getenv("SHINGLE_SIZE", "3"))

    # Jaccard similarity threshold for LSH band partitioning.
    # Pairs with estimated Jaccard >= this value are returned as candidates.
    lsh_threshold: float = float(os.getenv("LSH_THRESHOLD", "0.5"))

    # ── Semantic Verification ─────────────────────────────────────────────────
    # HuggingFace model name used for sentence embeddings.
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # Cosine similarity threshold above which a pair is labelled near-duplicate.
    semantic_threshold: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.85"))

    # ── Classification ────────────────────────────────────────────────────────
    # Jaccard threshold above which a pair is considered an exact duplicate
    # (before semantic check).
    duplicate_jaccard_threshold: float = float(
        os.getenv("DUPLICATE_JACCARD_THRESHOLD", "0.9")
    )

    # ── Storage ───────────────────────────────────────────────────────────────
    # SQLite database URL (can point to PostgreSQL in production).
    database_url: str = os.getenv(
        "DATABASE_URL", "sqlite:///novelty_detector.db"
    )

    # ── Dashboard ─────────────────────────────────────────────────────────────
    flask_secret_key: str = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
    flask_debug: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"


# Singleton instance — importable directly by other modules.
settings = Settings()
