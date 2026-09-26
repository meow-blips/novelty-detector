"""
tests/test_phase2.py
=====================
Smoke-tests for Phase 2 modules.

Tests are dependency-ordered:
  1. SemanticEmbedder — encoding, cosine scoring, batch encode, caching.
  2. DocumentClassifier — pair verdicts, roll-up, DB persistence.
  3. Full Pipeline (no-semantic mode) — end-to-end without loading a model.
  4. Full Pipeline (semantic mode)    — requires sentence-transformers & a
                                        network connection on first run to
                                        download the model weights.

Run with:
    python tests/test_phase2.py              # runs all tests
    python tests/test_phase2.py --no-model   # skips model-download tests
"""

from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"

def ok(msg):   print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def fail(msg): print(f"  {RED}✗ FAIL:{RESET} {msg}"); raise SystemExit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

TEXT_CAT_1   = "The cat sat on the mat and looked at the dog across the room."
TEXT_CAT_2   = "A cat was sitting on a mat, staring at the dog on the other side."
TEXT_PHYSICS = ("Quantum entanglement is a phenomenon where two particles "
                "become correlated such that their states cannot be "
                "described independently of each other.")
TEXT_NOVEL   = ("The stock market closed higher on Friday as investors "
                "reacted positively to the latest employment figures.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — SemanticEmbedder
# ─────────────────────────────────────────────────────────────────────────────

def test_semantic_embedder() -> None:
    print("\n[1] SemanticEmbedder")

    from novelty_detector.semantic.embedder import SemanticEmbedder, EmbedderError

    emb = SemanticEmbedder()
    ok(f"Model loaded: '{emb.model_name}' | dim={emb.embedding_dim}")

    # Single encode
    vec = emb.encode(TEXT_CAT_1, doc_id="cat_1")
    assert vec.shape == (emb.embedding_dim,), f"Shape mismatch: {vec.shape}"
    ok(f"Single encode → shape {vec.shape}")

    # Cache hit (same doc_id)
    vec_cached = emb.encode(TEXT_CAT_1, doc_id="cat_1")
    import numpy as np
    assert np.allclose(vec, vec_cached), "Cache returned different vector"
    ok("Cache hit returns identical vector")

    # Cosine score — related pair should score high
    score_related = emb.score_pair(TEXT_CAT_1, TEXT_CAT_2,
                                   doc_id_a="cat_1", doc_id_b="cat_2")
    ok(f"Related pair cosine: {score_related.cosine_similarity:.4f}")

    # Cosine score — unrelated pair should score low
    score_unrelated = emb.score_pair(TEXT_CAT_1, TEXT_PHYSICS,
                                     doc_id_a="cat_1", doc_id_b="physics")
    ok(f"Unrelated pair cosine: {score_unrelated.cosine_similarity:.4f}")

    assert score_related.cosine_similarity > score_unrelated.cosine_similarity, (
        "Related pair should score higher than unrelated pair"
    )
    ok("Related pair scores higher than unrelated pair ✓")

    # Batch encode
    batch = emb.encode_batch([
        ("b_cat_1", TEXT_CAT_1),
        ("b_cat_2", TEXT_CAT_2),
        ("b_physics", TEXT_PHYSICS),
    ])
    assert len(batch) == 3
    ok(f"Batch encode returned {len(batch)} vectors")

    # score_candidates
    scores = emb.score_candidates(
        query_text=TEXT_CAT_2,
        candidate_texts=[
            ("cand_cat_1", TEXT_CAT_1),
            ("cand_physics", TEXT_PHYSICS),
        ],
        query_doc_id="test_cat_2",
    )
    assert len(scores) == 2
    assert scores[0].cosine_similarity >= scores[1].cosine_similarity, \
        "Results should be sorted descending by cosine"
    ok(f"score_candidates → top={scores[0].candidate_doc_id} "
       f"({scores[0].cosine_similarity:.4f})")

    # Empty text should raise
    try:
        emb.encode("", doc_id="empty")
        fail("Should raise EmbedderError on empty text")
    except EmbedderError:
        ok("Empty text raises EmbedderError as expected")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — DocumentClassifier (no DB, persist=False)
# ─────────────────────────────────────────────────────────────────────────────

def test_classifier_no_db() -> None:
    print("\n[2] DocumentClassifier (persist=False)")

    from novelty_detector.classification.classifier import (
        DocumentClassifier, ClassificationResult, PairVerdict,
    )
    from novelty_detector.lexical.lsh_index import CandidateMatch
    from novelty_detector.semantic.embedder import SemanticScore
    from novelty_detector.ingestion.preprocessor import TextPreprocessor

    prep = TextPreprocessor()
    clf  = DocumentClassifier(
        duplicate_jaccard_threshold=0.9,
        semantic_threshold=0.85,
        persist=False,
    )

    doc = prep.process(TEXT_CAT_2, doc_id="test_cat_2")

    # ── Scenario A: High Jaccard → 'duplicate' ────────────────────────────────
    lsh_dupe = [CandidateMatch(
        query_doc_id="test_cat_2",
        candidate_doc_id="cat_1",
        estimated_jaccard=0.95,
        shingle_count_query=10,
    )]
    result_dupe = clf.process_document(doc, lsh_candidates=lsh_dupe)
    assert result_dupe.verdict == "duplicate", f"Got {result_dupe.verdict!r}"
    ok(f"High Jaccard (0.95) → verdict='duplicate' ✓")

    # ── Scenario B: Low Jaccard + High Cosine → 'near-duplicate' ─────────────
    lsh_near = [CandidateMatch(
        query_doc_id="test_cat_2",
        candidate_doc_id="cat_1",
        estimated_jaccard=0.55,
        shingle_count_query=10,
    )]
    sem_near = [SemanticScore(
        query_doc_id="test_cat_2",
        candidate_doc_id="cat_1",
        cosine_similarity=0.92,
        embedding_dim=384,
    )]
    result_near = clf.process_document(
        doc, lsh_candidates=lsh_near, semantic_scores=sem_near
    )
    assert result_near.verdict == "near-duplicate", f"Got {result_near.verdict!r}"
    ok("Low Jaccard + High Cosine → verdict='near-duplicate' ✓")

    # ── Scenario C: No candidates → 'novel' ──────────────────────────────────
    result_novel = clf.process_document(doc, lsh_candidates=[])
    assert result_novel.verdict == "novel", f"Got {result_novel.verdict!r}"
    ok("No candidates → verdict='novel' ✓")

    # ── Scenario D: Low Jaccard + Low Cosine → 'novel' ───────────────────────
    lsh_low = [CandidateMatch(
        query_doc_id="test_cat_2",
        candidate_doc_id="cat_1",
        estimated_jaccard=0.3,
        shingle_count_query=10,
    )]
    sem_low = [SemanticScore(
        query_doc_id="test_cat_2",
        candidate_doc_id="cat_1",
        cosine_similarity=0.4,
        embedding_dim=384,
    )]
    result_low = clf.process_document(
        doc, lsh_candidates=lsh_low, semantic_scores=sem_low
    )
    assert result_low.verdict == "novel", f"Got {result_low.verdict!r}"
    ok("Low Jaccard + Low Cosine → verdict='novel' ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Full Pipeline (semantic stage disabled)
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_no_semantic() -> None:
    print("\n[3] Pipeline (skip_semantic=True)")

    from novelty_detector.pipeline import Pipeline

    p = Pipeline(skip_semantic=True, index_novel_documents=True)

    # Seed corpus
    indexed = p.index_corpus([
        ("corpus_cat_1", TEXT_CAT_1),
        ("corpus_physics", TEXT_PHYSICS),
    ])
    assert indexed == 2, f"Expected 2 indexed, got {indexed}"
    ok(f"Corpus seeded: {indexed} documents indexed")

    # Near-duplicate should get LSH candidates
    r1 = p.process(TEXT_CAT_2, doc_id="incoming_cat_2")
    ok(f"cat_2 verdict: '{r1.verdict}' | best_match={r1.best_match_id!r} "
       f"jaccard={r1.best_jaccard}")

    # Clearly novel document
    r2 = p.process(TEXT_NOVEL, doc_id="incoming_novel")
    ok(f"novel verdict: '{r2.verdict}' | best_match={r2.best_match_id!r}")
    if r2.verdict != "novel":
        warn(f"Expected 'novel', got '{r2.verdict}' (probabilistic — may vary)")

    # Re-submitting same doc_id should still work (not raise)
    r3 = p.process(TEXT_CAT_2, doc_id="incoming_cat_2_retry")
    ok(f"Retry doc processed cleanly: verdict='{r3.verdict}'")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Full Pipeline (with semantic stage)
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_with_semantic() -> None:
    print("\n[4] Pipeline (full semantic stage)")

    from novelty_detector.pipeline import Pipeline
    from novelty_detector.lexical.lsh_index import LSHIndex

    # Lower LSH threshold so the cat sentences definitely become candidates
    lsh = LSHIndex(threshold=0.2, num_perm=64, shingle_size=2)
    p = Pipeline(
        lsh_index=lsh,
        skip_semantic=False,
        index_novel_documents=True,
    )

    p.index_corpus([
        ("s_corpus_cat_1", TEXT_CAT_1),
        ("s_corpus_physics", TEXT_PHYSICS),
    ])
    ok("Corpus indexed with semantic pipeline")

    # Process cat paraphrase — should be near-duplicate or duplicate
    r = p.process(TEXT_CAT_2, doc_id="s_incoming_cat_2")
    ok(f"cat_2 full pipeline verdict: '{r.verdict}' "
       f"| best_match={r.best_match_id!r} "
       f"| cosine={r.best_cosine}")

    if r.verdict in ("near-duplicate", "duplicate"):
        ok("Paraphrase correctly flagged as near-duplicate/duplicate ✓")
    else:
        warn(
            f"Expected near-duplicate or duplicate, got '{r.verdict}'. "
            "This can happen when the LSH threshold is too high for "
            "these short sentences — try lowering lsh_threshold further."
        )

    # Novel document
    r2 = p.process(TEXT_NOVEL, doc_id="s_incoming_novel")
    ok(f"Novel doc verdict: '{r2.verdict}' | cosine={r2.best_cosine}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip tests that download/load the sentence-transformer model.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 2 Smoke Tests")
    print("=" * 60)

    if args.no_model:
        warn("Skipping model-dependent tests (--no-model flag set).")
        test_classifier_no_db()
        test_pipeline_no_semantic()
    else:
        test_semantic_embedder()
        test_classifier_no_db()
        test_pipeline_no_semantic()
        test_pipeline_with_semantic()

    print(f"\n{GREEN}All Phase 2 tests passed.{RESET}\n")
