"""
novelty_detector/dashboard/routes.py
=======================================
All Flask Routes — Reviewer Dashboard (Phase 3)
-------------------------------------------------
Routes are grouped in a single Blueprint ``bp``.

  GET  /                       → Overview / stats dashboard
  GET  /documents              → All ingested documents (paginated)
  GET  /flagged                → All flagged pairs (near-dup / duplicate)
  GET  /review/<int:cmp_id>    → Side-by-side review of one pair
  POST /review/<int:cmp_id>    → Submit human override / confirm
  GET  /settings               → Threshold configuration page
  POST /settings               → Apply new thresholds
  GET  /submit                 → Document submission form
  POST /submit                 → Run pipeline + redirect to review if flagged
  GET  /api/stats              → JSON stats (for dashboard charts via fetch())
"""

from __future__ import annotations

import difflib
from typing import List

from flask import (
    Blueprint,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from loguru import logger
from sqlalchemy import func

from novelty_detector.classification.classifier import DocumentClassifier
from novelty_detector.config import settings
from novelty_detector.dashboard.forms import (
    ReviewDecisionForm,
    SubmitDocumentForm,
    ThresholdSettingsForm,
)
from novelty_detector.ingestion.preprocessor import PreprocessingError, TextPreprocessor
from novelty_detector.lexical.lsh_index import LSHIndex
from novelty_detector.pipeline import Pipeline, PipelineError
from novelty_detector.semantic.embedder import SemanticEmbedder
from novelty_detector.storage.database import get_session
from novelty_detector.storage.models import ComparisonLog, DocumentRecord

bp = Blueprint("dashboard", __name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy Pipeline singleton
# The pipeline is expensive to construct (loads the embedding model).
# We build it once on first request and cache it in module scope.
# ─────────────────────────────────────────────────────────────────────────────

_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    """Return (or lazily create) the shared Pipeline instance."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising pipeline for dashboard…")
        _pipeline = Pipeline(
            lsh_index=LSHIndex(
                threshold=settings.lsh_threshold,
                num_perm=settings.minhash_num_perm,
                shingle_size=settings.shingle_size,
            ),
            skip_semantic=False,
            index_novel_documents=True,
        )
        # Warm-up: re-index all existing DocumentRecords so the LSH is
        # populated from the DB on server restart.
        _warm_up_index(_pipeline)
    return _pipeline


def _warm_up_index(pipeline: Pipeline) -> None:
    """Re-index all stored DocumentRecords into the in-memory LSH on startup."""
    try:
        with get_session() as session:
            records = session.query(DocumentRecord).all()
        if not records:
            return
        corpus = [
            (r.doc_id, r.clean_text or r.raw_text or "")
            for r in records
            if (r.clean_text or r.raw_text)
        ]
        if corpus:
            pipeline.index_corpus(corpus)
            logger.info("Warm-up: re-indexed {} documents into LSH.", len(corpus))
    except Exception as exc:
        logger.warning("LSH warm-up failed (non-fatal): {}", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _diff_html(text_a: str, text_b: str) -> tuple[str, str]:
    """
    Generate word-level HTML diff for two texts.

    Returns
    -------
    Tuple[str, str]
        (html_a, html_b) — the two sides with <mark> tags highlighting
        added / removed words.
    """
    words_a = text_a.split()
    words_b = text_b.split()

    matcher = difflib.SequenceMatcher(None, words_a, words_b)
    parts_a: List[str] = []
    parts_b: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        chunk_a = " ".join(words_a[i1:i2])
        chunk_b = " ".join(words_b[j1:j2])
        if tag == "equal":
            parts_a.append(chunk_a)
            parts_b.append(chunk_b)
        elif tag == "replace":
            parts_a.append(f'<mark class="diff-del">{chunk_a}</mark>')
            parts_b.append(f'<mark class="diff-ins">{chunk_b}</mark>')
        elif tag == "delete":
            parts_a.append(f'<mark class="diff-del">{chunk_a}</mark>')
        elif tag == "insert":
            parts_b.append(f'<mark class="diff-ins">{chunk_b}</mark>')

    return " ".join(parts_a), " ".join(parts_b)


def _paginate(query, page: int, per_page: int = 20):
    """Return a SQLAlchemy paginated result."""
    return query.paginate(page=page, per_page=per_page, error_out=False)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@bp.route("/")
def index():
    """Overview dashboard with high-level statistics."""
    with get_session() as session:
        total_docs = session.query(func.count(DocumentRecord.id)).scalar() or 0
        novel_count = (
            session.query(func.count(DocumentRecord.id))
            .filter(DocumentRecord.status == "novel")
            .scalar() or 0
        )
        near_dup_count = (
            session.query(func.count(DocumentRecord.id))
            .filter(DocumentRecord.status == "near-duplicate")
            .scalar() or 0
        )
        dup_count = (
            session.query(func.count(DocumentRecord.id))
            .filter(DocumentRecord.status == "duplicate")
            .scalar() or 0
        )
        pending_reviews = (
            session.query(func.count(ComparisonLog.id))
            .filter(ComparisonLog.review_status == "pending")
            .filter(ComparisonLog.verdict != "novel")
            .scalar() or 0
        )
        # Most recent 5 flagged comparisons for the "Recent Activity" feed
        recent_flags = (
            session.query(ComparisonLog)
            .filter(ComparisonLog.verdict != "novel")
            .order_by(ComparisonLog.compared_at.desc())
            .limit(5)
            .all()
        )

    return render_template(
        "index.html",
        total_docs=total_docs,
        novel_count=novel_count,
        near_dup_count=near_dup_count,
        dup_count=dup_count,
        pending_reviews=pending_reviews,
        recent_flags=recent_flags,
        settings=settings,
    )


@bp.route("/documents")
def documents():
    """Paginated list of all ingested documents with their status."""
    page = request.args.get("page", 1, type=int)
    status_filter = request.args.get("status", "")
    search = request.args.get("q", "").strip()

    with get_session() as session:
        q = session.query(DocumentRecord).order_by(
            DocumentRecord.submitted_at.desc()
        )
        if status_filter:
            q = q.filter(DocumentRecord.status == status_filter)
        if search:
            q = q.filter(DocumentRecord.doc_id.ilike(f"%{search}%"))

        pagination = q.paginate(page=page, per_page=25, error_out=False)
        records = pagination.items

    return render_template(
        "documents.html",
        records=records,
        pagination=pagination,
        status_filter=status_filter,
        search=search,
    )


@bp.route("/flagged")
def flagged():
    """All flagged comparison pairs (near-duplicate or duplicate)."""
    page = request.args.get("page", 1, type=int)
    verdict_filter = request.args.get("verdict", "")
    review_filter = request.args.get("review", "")

    with get_session() as session:
        q = (
            session.query(ComparisonLog)
            .filter(ComparisonLog.verdict != "novel")
            .order_by(ComparisonLog.compared_at.desc())
        )
        if verdict_filter:
            q = q.filter(ComparisonLog.verdict == verdict_filter)
        if review_filter:
            q = q.filter(ComparisonLog.review_status == review_filter)

        pagination = q.paginate(page=page, per_page=20, error_out=False)
        comparisons = pagination.items

    return render_template(
        "flagged.html",
        comparisons=comparisons,
        pagination=pagination,
        verdict_filter=verdict_filter,
        review_filter=review_filter,
    )


@bp.route("/review/<int:cmp_id>", methods=["GET", "POST"])
def review(cmp_id: int):
    """Side-by-side document review page with confirm / override form."""
    form = ReviewDecisionForm()

    with get_session() as session:
        comparison = session.query(ComparisonLog).get(cmp_id)
        if comparison is None:
            flash("Comparison not found.", "danger")
            return redirect(url_for("dashboard.flagged"))

        query_doc = (
            session.query(DocumentRecord)
            .filter_by(doc_id=comparison.query_doc_id)
            .first()
        )
        candidate_doc = (
            session.query(DocumentRecord)
            .filter_by(doc_id=comparison.candidate_doc_id)
            .first()
        )

        if form.validate_on_submit():
            # ── Process reviewer decision ──────────────────────────────────────
            decision = form.decision.data
            comparison.review_status = decision
            comparison.reviewer_note = form.reviewer_note.data or None

            if decision == "overridden" and form.new_verdict.data:
                new_verdict = form.new_verdict.data
                comparison.verdict = new_verdict
                # Also update the query document's status if it was downgraded.
                if query_doc:
                    query_doc.status = new_verdict
                    query_doc.reviewed = True
                logger.info(
                    "Review override: pair ({!r}, {!r}) → {}",
                    comparison.query_doc_id,
                    comparison.candidate_doc_id,
                    new_verdict,
                )
            else:
                if query_doc:
                    query_doc.reviewed = True
                logger.info(
                    "Review confirmed: pair ({!r}, {!r}) verdict={}",
                    comparison.query_doc_id,
                    comparison.candidate_doc_id,
                    comparison.verdict,
                )

            session.commit()
            flash("Review saved successfully.", "success")
            return redirect(url_for("dashboard.flagged"))

        # ── Populate form defaults on GET ──────────────────────────────────────
        form.comparison_id.data = str(cmp_id)
        form.decision.data = comparison.review_status

        # Word-level diff
        text_a = query_doc.raw_text if query_doc else "(text not available)"
        text_b = candidate_doc.raw_text if candidate_doc else "(text not available)"
        diff_a, diff_b = _diff_html(text_a, text_b)

    return render_template(
        "review.html",
        comparison=comparison,
        query_doc=query_doc,
        candidate_doc=candidate_doc,
        diff_a=diff_a,
        diff_b=diff_b,
        form=form,
    )


@bp.route("/settings", methods=["GET", "POST"])
def threshold_settings():
    """Live threshold adjustment. Changes are in-process only."""
    form = ThresholdSettingsForm()

    if form.validate_on_submit():
        # Mutate the settings singleton (in-memory, not persisted to .env).
        old = {
            "lsh": settings.lsh_threshold,
            "semantic": settings.semantic_threshold,
            "dup_jaccard": settings.duplicate_jaccard_threshold,
        }
        settings.lsh_threshold = form.lsh_threshold.data
        settings.semantic_threshold = form.semantic_threshold.data
        settings.duplicate_jaccard_threshold = form.duplicate_jaccard_threshold.data

        # Invalidate the cached pipeline so a new LSHIndex is built with
        # the updated threshold on next request.
        global _pipeline
        _pipeline = None

        logger.info(
            "Thresholds updated: {} → lsh={} semantic={} dup_jaccard={}",
            old,
            settings.lsh_threshold,
            settings.semantic_threshold,
            settings.duplicate_jaccard_threshold,
        )
        flash(
            "Thresholds updated. The pipeline index has been reset — "
            "re-index your corpus before processing new documents.",
            "warning",
        )
        return redirect(url_for("dashboard.threshold_settings"))

    # Pre-fill form with current settings.
    if request.method == "GET":
        form.lsh_threshold.data = settings.lsh_threshold
        form.semantic_threshold.data = settings.semantic_threshold
        form.duplicate_jaccard_threshold.data = settings.duplicate_jaccard_threshold

    return render_template("settings.html", form=form, settings=settings)


@bp.route("/submit", methods=["GET", "POST"])
def submit_document():
    """Submit a document through the full pipeline from the browser UI."""
    form = SubmitDocumentForm()

    if form.validate_on_submit():
        doc_id = form.doc_id.data.strip()
        raw_text = form.raw_text.data.strip()

        try:
            pipeline = _get_pipeline()
            result = pipeline.process(raw_text, doc_id=doc_id)
        except PipelineError as exc:
            flash(f"Pipeline error: {exc}", "danger")
            return render_template("submit.html", form=form)
        except Exception as exc:
            logger.exception("Unexpected error processing doc_id={!r}", doc_id)
            flash(f"Unexpected error: {exc}", "danger")
            return render_template("submit.html", form=form)

        verdict = result.verdict
        flash(
            f"Document '{doc_id}' processed. Verdict: {verdict.upper()}",
            "success" if verdict == "novel" else "warning",
        )

        # If flagged, redirect to the most recent comparison log for this doc.
        if verdict in ("near-duplicate", "duplicate"):
            with get_session() as session:
                log = (
                    session.query(ComparisonLog)
                    .filter_by(query_doc_id=doc_id)
                    .order_by(ComparisonLog.compared_at.desc())
                    .first()
                )
            if log:
                return redirect(url_for("dashboard.review", cmp_id=log.id))

        return redirect(url_for("dashboard.index"))

    return render_template("submit.html", form=form)


# ─────────────────────────────────────────────────────────────────────────────
# JSON API (for the front-end chart / polling)
# ─────────────────────────────────────────────────────────────────────────────

@bp.route("/api/stats")
def api_stats():
    """Return dashboard statistics as JSON for front-end fetch() calls."""
    with get_session() as session:
        totals = (
            session.query(DocumentRecord.status, func.count(DocumentRecord.id))
            .group_by(DocumentRecord.status)
            .all()
        )
        pending = (
            session.query(func.count(ComparisonLog.id))
            .filter(ComparisonLog.review_status == "pending")
            .filter(ComparisonLog.verdict != "novel")
            .scalar() or 0
        )

    status_counts = {row[0]: row[1] for row in totals}
    return jsonify(
        {
            "novel": status_counts.get("novel", 0),
            "near_duplicate": status_counts.get("near-duplicate", 0),
            "duplicate": status_counts.get("duplicate", 0),
            "pending_reviews": pending,
            "thresholds": {
                "lsh": settings.lsh_threshold,
                "semantic": settings.semantic_threshold,
                "duplicate_jaccard": settings.duplicate_jaccard_threshold,
            },
        }
    )
