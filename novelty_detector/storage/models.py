"""
novelty_detector/storage/models.py
=====================================
SQLAlchemy ORM Models — Audit Log & Document Registry
------------------------------------------------------
This module defines the database schema for persisting:

  1. ``DocumentRecord``  — every document ever submitted for analysis.
  2. ``ComparisonLog``   — the detailed result for every flagged pair,
                           including raw Jaccard, cosine similarity, final
                           verdict, and optional human override.

The schema deliberately keeps things simple (two tables) so SQLite works
in development and PostgreSQL can be swapped in by changing ``DATABASE_URL``.

Alembic is the migration tool (see ``alembic/`` directory, added in Phase 3).
For Phase 2 we call ``Base.metadata.create_all(engine)`` on startup.

Usage
-----
    from novelty_detector.storage.database import get_session
    from novelty_detector.storage.models import DocumentRecord, ComparisonLog

    with get_session() as session:
        session.add(DocumentRecord(doc_id="doc_1", status="novel"))
        session.commit()
"""

from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ─────────────────────────────────────────────────────────────────────────────
# SQLAlchemy 2.x declarative base
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Shared base class for all ORM models."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Enums (stored as VARCHAR for portability; no DB-level enum type needed)
# ─────────────────────────────────────────────────────────────────────────────

#: Valid classification verdicts.
VERDICT_ENUM = ("novel", "near-duplicate", "duplicate")

#: Valid human review decisions (override choices).
REVIEW_ENUM = ("confirmed", "overridden", "pending")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 — DocumentRecord
# ─────────────────────────────────────────────────────────────────────────────

class DocumentRecord(Base):
    """
    One row per document submitted for analysis.

    Columns
    -------
    id              Auto-increment primary key.
    doc_id          Caller-supplied stable identifier (filename, UUID, …).
    submitted_at    UTC timestamp of ingestion.
    raw_text        The original raw text (stored for dashboard display).
    clean_text      The normalised, stemmed text (for re-embedding if needed).
    token_count     Number of tokens after preprocessing.
    status          Final verdict: ``novel`` | ``near-duplicate`` | ``duplicate``.
    reviewed        Whether a human reviewer has looked at this document.
    """

    __tablename__ = "document_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    doc_id: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        unique=True,
        index=True,
        comment="Caller-supplied stable document identifier.",
    )

    submitted_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
        comment="UTC timestamp when the document was ingested.",
    )

    raw_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Original unmodified text; stored for dashboard side-by-side view.",
    )

    clean_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Preprocessed, stemmed text used for similarity computations.",
    )

    token_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of tokens after preprocessing.",
    )

    status: Mapped[str] = mapped_column(
        Enum(*VERDICT_ENUM, name="verdict_enum"),
        nullable=False,
        default="novel",
        index=True,
        comment="System classification: novel | near-duplicate | duplicate.",
    )

    reviewed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if a human reviewer has confirmed or overridden the verdict.",
    )

    # One-to-many: this document as the *query* side of a comparison.
    comparisons_as_query: Mapped[list["ComparisonLog"]] = relationship(
        "ComparisonLog",
        foreign_keys="ComparisonLog.query_doc_id",
        back_populates="query_document",
        cascade="all, delete-orphan",
    )

    # One-to-many: this document as the *candidate* side.
    comparisons_as_candidate: Mapped[list["ComparisonLog"]] = relationship(
        "ComparisonLog",
        foreign_keys="ComparisonLog.candidate_doc_id",
        back_populates="candidate_document",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentRecord doc_id={self.doc_id!r} status={self.status!r}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 — ComparisonLog
# ─────────────────────────────────────────────────────────────────────────────

class ComparisonLog(Base):
    """
    One row per document **pair** that passed the LSH threshold (Stage 1)
    and was then scored semantically (Stage 2).

    Columns
    -------
    id                  Auto-increment primary key.
    query_doc_id        FK → DocumentRecord.doc_id (the incoming document).
    candidate_doc_id    FK → DocumentRecord.doc_id (the matched document).
    compared_at         UTC timestamp of the comparison.
    jaccard_estimate    MinHash-estimated Jaccard similarity from Stage 1.
    cosine_similarity   Cosine similarity from Stage 2 (None if skipped).
    verdict             System classification for this pair.
    lsh_threshold_used  The LSH threshold that was active when the pair was flagged.
    semantic_threshold  The cosine threshold used for classification.
    review_status       Human review decision: confirmed | overridden | pending.
    reviewer_note       Free-text note from the human reviewer.
    """

    __tablename__ = "comparison_logs"

    # ── Ensure each ordered pair is logged at most once ───────────────────────
    __table_args__ = (
        UniqueConstraint(
            "query_doc_id", "candidate_doc_id",
            name="uq_pair",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # FK references doc_id (string), not the integer PK, for readability in logs.
    query_doc_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("document_records.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    candidate_doc_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("document_records.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    compared_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.datetime.utcnow,
    )

    # Stage 1 score (always present — the LSH filter provides it)
    jaccard_estimate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="MinHash-estimated Jaccard similarity from Stage 1.",
    )

    # Stage 2 score (None if the Jaccard alone was enough to classify as duplicate)
    cosine_similarity: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Cosine similarity from sentence embeddings (Stage 2).",
    )

    verdict: Mapped[str] = mapped_column(
        Enum(*VERDICT_ENUM, name="verdict_enum"),
        nullable=False,
        index=True,
    )

    # Thresholds active at comparison time (useful for retroactive analysis
    # when an admin adjusts thresholds on the dashboard).
    lsh_threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    semantic_threshold_used: Mapped[float] = mapped_column(Float, nullable=False)

    review_status: Mapped[str] = mapped_column(
        Enum(*REVIEW_ENUM, name="review_enum"),
        nullable=False,
        default="pending",
        index=True,
    )

    reviewer_note: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Free-text note from the human reviewer.",
    )

    # ORM relationships
    query_document: Mapped["DocumentRecord"] = relationship(
        "DocumentRecord",
        foreign_keys=[query_doc_id],
        back_populates="comparisons_as_query",
    )

    candidate_document: Mapped["DocumentRecord"] = relationship(
        "DocumentRecord",
        foreign_keys=[candidate_doc_id],
        back_populates="comparisons_as_candidate",
    )

    def __repr__(self) -> str:
        return (
            f"<ComparisonLog "
            f"query={self.query_doc_id!r} "
            f"candidate={self.candidate_doc_id!r} "
            f"verdict={self.verdict!r} "
            f"cosine={self.cosine_similarity}>"
        )
