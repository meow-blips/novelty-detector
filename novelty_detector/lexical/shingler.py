"""
novelty_detector/lexical/shingler.py
======================================
N-gram Shingling Utility (Phase 1 — Lexical Filter)
-----------------------------------------------------
Converts a list of tokens into a set of overlapping n-gram shingles.

A *shingle* (or k-gram) is a contiguous sequence of ``n`` tokens drawn from
a token stream.  When represented as a set, shingles allow Jaccard similarity
to be estimated efficiently via MinHash.

Why token-based shingles?
  - Character-level shingles (e.g., "the qu") are sensitive to minor
    typos but ignore word-level order.
  - Token-level shingles (e.g., ("the", "quick", "brown")) strike a good
    balance: they capture local phrasing while being robust to minor
    character variations after normalisation.

Example
-------
    tokens = ["quick", "brown", "fox", "jump", "over", "lazi", "dog"]
    shingles = build_shingles(tokens, n=3)
    # → {"quick brown fox", "brown fox jump", "fox jump over",
    #    "jump over lazi", "over lazi dog"}
"""

from __future__ import annotations

from typing import List, Set

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class ShinglingError(Exception):
    """Raised when shingling cannot produce a valid shingle set."""


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_shingles(tokens: List[str], n: int = 3) -> Set[str]:
    """
    Build a set of overlapping token n-gram shingles.

    Parameters
    ----------
    tokens : List[str]
        Pre-processed token list (output of ``TextPreprocessor.process()``).
    n : int
        Shingle size (number of tokens per shingle). Default: 3.

    Returns
    -------
    Set[str]
        A set of space-joined n-gram strings.

    Raises
    ------
    ShinglingError
        If ``n < 1``, or if the token list is too short to form even one
        shingle (len(tokens) < n).

    Notes
    -----
    - The returned set may be empty if ``tokens`` is shorter than ``n`` —
      callers should guard against empty shingle sets before building MinHash.
    - Using a ``set`` deduplicates repeated n-grams, which is intentional:
      Jaccard similarity operates on sets, not multisets.
    """
    if n < 1:
        raise ShinglingError(f"Shingle size n must be >= 1, got {n}.")

    if not tokens:
        raise ShinglingError("Cannot shingle an empty token list.")

    if len(tokens) < n:
        # Soft warning: we still return a single shingle (the whole token list)
        # so the document can participate in LSH at a reduced accuracy.
        logger.warning(
            "Token list length ({}) < shingle size ({}). "
            "Returning a single shingle containing all tokens.",
            len(tokens),
            n,
        )
        return {" ".join(tokens)}

    # Sliding window: for tokens=[a,b,c,d] and n=3
    # → [(a,b,c), (b,c,d)]
    shingle_set: Set[str] = set()
    for i in range(len(tokens) - n + 1):
        shingle = " ".join(tokens[i : i + n])
        shingle_set.add(shingle)

    logger.debug(
        "Built {} shingles (n={}) from {} tokens.",
        len(shingle_set),
        n,
        len(tokens),
    )
    return shingle_set
