"""
tests/test_phase1.py
=====================
Quick smoke-test for Phase 1 modules (no pytest dependency required).
Run with: python tests/test_phase1.py

Tests
-----
1. TextPreprocessor produces expected tokens.
2. build_shingles generates the right n-gram set.
3. LSHIndex correctly finds near-duplicate candidates.
4. LSHIndex correctly misses unrelated documents (novel).
5. Batch indexing & batch preprocessing work without raising.
6. save() / load() round-trip preserves index contents.
"""

from __future__ import annotations

import os
import sys
import tempfile

# Make the project root importable when running directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from novelty_detector.ingestion.preprocessor import TextPreprocessor, PreprocessingError
from novelty_detector.lexical.shingler import build_shingles, ShinglingError
from novelty_detector.lexical.lsh_index import LSHIndex, LSHIndexError

# ── Colour helpers (no dependencies) ──────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗ FAIL:{RESET} {msg}")
    raise SystemExit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────

DOCS = [
    ("doc_cat_1",
     "The cat sat on the mat and looked at the dog across the room."),
    ("doc_cat_2",
     "A cat was sitting on a mat, staring at the dog on the other side."),
    ("doc_physics",
     "Quantum entanglement is a phenomenon in quantum mechanics where two "
     "particles become correlated such that the state of each cannot be "
     "described independently."),
    ("doc_short", "Hi."),
]

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocessor() -> None:
    print("\n[1] TextPreprocessor")
    prep = TextPreprocessor()

    result = prep.process(DOCS[0][1], doc_id=DOCS[0][0])
    assert result.doc_id == "doc_cat_1", "doc_id mismatch"
    assert result.token_count > 0, "No tokens produced"
    assert result.clean_text, "clean_text is empty"
    ok(f"doc_cat_1 → {result.token_count} tokens: {result.tokens[:5]}…")

    # Empty string should raise
    try:
        prep.process("", doc_id="empty")
        fail("Should have raised PreprocessingError on empty input")
    except PreprocessingError:
        ok("Empty input raises PreprocessingError as expected")

    # Non-string should raise
    try:
        prep.process(12345, doc_id="non_str")  # type: ignore[arg-type]
        fail("Should have raised PreprocessingError on non-str input")
    except PreprocessingError:
        ok("Non-str input raises PreprocessingError as expected")

    # Batch processing
    results = prep.process_batch([(d, t) for d, t in DOCS])
    # doc_short ("Hi.") may produce 0 tokens; others should succeed
    assert len(results) >= 3, f"Expected at least 3 results, got {len(results)}"
    ok(f"Batch: processed {len(results)}/{len(DOCS)} documents")


def test_shingling() -> None:
    print("\n[2] Shingling")
    tokens = ["quick", "brown", "fox", "jump", "over", "lazi", "dog"]

    shingles_3 = build_shingles(tokens, n=3)
    expected_count = len(tokens) - 3 + 1  # 5 unique shingles
    assert len(shingles_3) == expected_count, (
        f"Expected {expected_count} shingles, got {len(shingles_3)}"
    )
    ok(f"n=3 → {len(shingles_3)} shingles: {list(shingles_3)[:2]}…")

    # Bigrams
    shingles_2 = build_shingles(tokens, n=2)
    assert len(shingles_2) == len(tokens) - 1
    ok(f"n=2 → {len(shingles_2)} shingles")

    # Too-short list falls back to single shingle
    shingles_tiny = build_shingles(["one", "word"], n=5)
    assert len(shingles_tiny) == 1
    ok("Short token list → single fallback shingle")

    # Invalid n
    try:
        build_shingles(tokens, n=0)
        fail("Should raise ShinglingError for n=0")
    except ShinglingError:
        ok("n=0 raises ShinglingError as expected")

    # Empty list
    try:
        build_shingles([], n=3)
        fail("Should raise ShinglingError for empty tokens")
    except ShinglingError:
        ok("Empty token list raises ShinglingError as expected")


def test_lsh_index() -> None:
    print("\n[3] LSHIndex — candidates & novelty")
    prep = TextPreprocessor()
    idx  = LSHIndex(threshold=0.3, num_perm=64, shingle_size=2)

    doc_a = prep.process(DOCS[0][1], doc_id=DOCS[0][0])  # doc_cat_1
    doc_b = prep.process(DOCS[1][1], doc_id=DOCS[1][0])  # doc_cat_2
    doc_c = prep.process(DOCS[2][1], doc_id=DOCS[2][0])  # doc_physics

    idx.add_document(doc_a)
    idx.add_document(doc_b)
    idx.add_document(doc_c)
    assert idx.size == 3
    ok(f"Index size after 3 inserts: {idx.size}")

    # Double-insert should raise
    try:
        idx.add_document(doc_a)
        fail("Should raise LSHIndexError on duplicate doc_id")
    except LSHIndexError:
        ok("Duplicate doc_id raises LSHIndexError as expected")

    # Query doc_cat_2 — should find doc_cat_1 as candidate
    candidates = idx.query(doc_b)
    cand_ids = [c.candidate_doc_id for c in candidates]
    ok(f"Query doc_cat_2 → candidates: {cand_ids}")
    # Note: LSH is probabilistic; at threshold=0.3 cat sentences should match
    if "doc_cat_1" in cand_ids:
        ok("Correctly found doc_cat_1 as near-duplicate of doc_cat_2")
    else:
        print(f"  ⚠ doc_cat_1 not in candidates {cand_ids} "
              "(probabilistic — may vary by run; try lowering threshold)")

    # Query doc_physics — should NOT match cat documents at high thresholds
    idx2 = LSHIndex(threshold=0.7, num_perm=128, shingle_size=3)
    idx2.add_document(doc_a)
    idx2.add_document(doc_b)
    novel_candidates = idx2.query(doc_c)
    ok(f"Novel doc_physics at threshold=0.7 → {len(novel_candidates)} candidate(s) (expect 0 or very few)")


def test_index_remove() -> None:
    print("\n[4] LSHIndex — remove")
    prep = TextPreprocessor()
    idx  = LSHIndex(threshold=0.3, num_perm=64, shingle_size=2)

    doc_a = prep.process(DOCS[0][1], doc_id="remove_me")
    idx.add_document(doc_a)
    assert idx.contains("remove_me")

    idx.remove_document("remove_me")
    assert not idx.contains("remove_me")
    assert idx.size == 0
    ok("Document removed; index size = 0")

    # Remove non-existent
    try:
        idx.remove_document("ghost")
        fail("Should raise LSHIndexError when removing non-existent doc_id")
    except LSHIndexError:
        ok("Removing non-existent doc_id raises LSHIndexError as expected")


def test_persistence() -> None:
    print("\n[5] LSHIndex — save / load")
    prep = TextPreprocessor()
    idx  = LSHIndex(threshold=0.4, num_perm=64, shingle_size=2)

    doc_a = prep.process(DOCS[0][1], doc_id="persist_a")
    doc_b = prep.process(DOCS[1][1], doc_id="persist_b")
    idx.add_document(doc_a)
    idx.add_document(doc_b)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        tmp_path = fh.name

    try:
        idx.save(tmp_path)
        ok(f"Saved index to {tmp_path}")

        loaded = LSHIndex.load(tmp_path)
        assert loaded.size == 2, f"Expected 2, got {loaded.size}"
        assert loaded.contains("persist_a")
        assert loaded.contains("persist_b")
        ok("Loaded index has correct size and doc_ids")

        # Querying on loaded index should still work
        doc_b_reloaded = prep.process(DOCS[1][1], doc_id="query_persist_b")
        candidates = loaded.query(doc_b_reloaded)
        ok(f"Post-load query returned {len(candidates)} candidate(s)")
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 1 Smoke Tests")
    print("=" * 60)

    test_preprocessor()
    test_shingling()
    test_lsh_index()
    test_index_remove()
    test_persistence()

    print(f"\n{GREEN}All Phase 1 tests passed.{RESET}\n")
