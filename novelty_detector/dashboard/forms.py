"""
novelty_detector/dashboard/forms.py
=====================================
Flask-WTF Forms for the Reviewer Dashboard
--------------------------------------------
All user-facing forms live here so validation logic is centralised
and CSRF protection is applied automatically by Flask-WTF.
"""

from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import (
    FloatField,
    StringField,
    SubmitField,
    TextAreaField,
    SelectField,
    HiddenField,
)
from wtforms.validators import (
    DataRequired,
    Length,
    NumberRange,
    Optional,
)


class SubmitDocumentForm(FlaskForm):
    """
    Form for submitting a new document to the pipeline via the dashboard UI.
    """
    doc_id = StringField(
        "Document ID",
        validators=[
            DataRequired(message="A unique document identifier is required."),
            Length(min=1, max=512, message="ID must be 1–512 characters."),
        ],
        render_kw={"placeholder": "e.g. paper_001 or a UUID"},
    )
    raw_text = TextAreaField(
        "Document Text",
        validators=[
            DataRequired(message="Document text cannot be empty."),
            Length(min=10, message="Text must be at least 10 characters."),
        ],
        render_kw={"rows": 12, "placeholder": "Paste your document text here…"},
    )
    submit = SubmitField("Analyse Document")


class ReviewDecisionForm(FlaskForm):
    """
    Form for a human reviewer to confirm or override a system verdict.
    Rendered on the side-by-side review page.
    """
    comparison_id = HiddenField("Comparison ID", validators=[DataRequired()])

    decision = SelectField(
        "Your Decision",
        choices=[
            ("confirmed", "✅  Confirm — system verdict is correct"),
            ("overridden", "🔄  Override — change the verdict"),
        ],
        validators=[DataRequired()],
    )

    new_verdict = SelectField(
        "Override Verdict (if overriding)",
        choices=[
            ("", "— select —"),
            ("novel", "Novel"),
            ("near-duplicate", "Near-Duplicate"),
            ("duplicate", "Duplicate"),
        ],
        validators=[Optional()],
    )

    reviewer_note = TextAreaField(
        "Reviewer Note",
        validators=[
            Optional(),
            Length(max=2000, message="Note must be under 2000 characters."),
        ],
        render_kw={
            "rows": 4,
            "placeholder": "Optional: explain your decision…",
        },
    )

    submit = SubmitField("Save Review")


class ThresholdSettingsForm(FlaskForm):
    """
    Form for adjusting the pipeline's similarity thresholds live via the UI.
    Changes are written to the ``settings`` singleton (in-process only;
    they are NOT persisted to ``.env`` automatically — a warning is shown).
    """
    lsh_threshold = FloatField(
        "LSH Jaccard Threshold (Stage 1)",
        validators=[
            DataRequired(),
            NumberRange(min=0.01, max=0.99,
                        message="Must be between 0.01 and 0.99"),
        ],
        render_kw={"step": "0.05", "min": "0.01", "max": "0.99"},
        description=(
            "Pairs with estimated Jaccard ≥ this value are passed to Stage 2."
        ),
    )
    semantic_threshold = FloatField(
        "Semantic Cosine Threshold (Stage 2)",
        validators=[
            DataRequired(),
            NumberRange(min=0.01, max=0.99,
                        message="Must be between 0.01 and 0.99"),
        ],
        render_kw={"step": "0.05", "min": "0.01", "max": "0.99"},
        description=(
            "Pairs with cosine similarity ≥ this value are labelled near-duplicate."
        ),
    )
    duplicate_jaccard_threshold = FloatField(
        "Duplicate Jaccard Threshold (exact copy cutoff)",
        validators=[
            DataRequired(),
            NumberRange(min=0.01, max=1.0,
                        message="Must be between 0.01 and 1.0"),
        ],
        render_kw={"step": "0.05", "min": "0.01", "max": "1.0"},
        description=(
            "Pairs with Jaccard ≥ this are immediately labelled duplicate "
            "(skips Stage 2)."
        ),
    )
    submit = SubmitField("Apply Thresholds")
