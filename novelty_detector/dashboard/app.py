"""
novelty_detector/dashboard/app.py
===================================
Flask Application Factory (Phase 3)
--------------------------------------
Creates and configures the Flask app instance.

Using the Application Factory pattern means:
- The app can be created multiple times (useful for testing).
- Extensions are initialised in a predictable order.
- The app object is never a module-level global (avoids circular imports).

Usage
-----
    # From the project root:
    python run.py

    # Or via Flask CLI:
    set FLASK_APP=novelty_detector.dashboard.app:create_app
    flask run --debug
"""

from __future__ import annotations

from flask import Flask
from loguru import logger

from novelty_detector.config import settings
from novelty_detector.storage.database import init_db


def create_app() -> Flask:
    """
    Construct and return a configured Flask application.

    Steps
    -----
    1. Create the Flask instance with correct template / static paths.
    2. Apply config from ``settings``.
    3. Initialise the database (create tables if absent).
    4. Register Blueprints (routes).
    5. Register custom Jinja2 filters.

    Returns
    -------
    Flask
        A fully configured, ready-to-run Flask application.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # ── Flask config ──────────────────────────────────────────────────────────
    app.config["SECRET_KEY"] = settings.flask_secret_key
    app.config["DEBUG"] = settings.flask_debug
    # Disable JSON key sorting so API responses preserve insertion order.
    app.config["JSON_SORT_KEYS"] = False

    # ── Database ──────────────────────────────────────────────────────────────
    init_db()
    logger.info("Database initialised.")

    # ── Blueprints ────────────────────────────────────────────────────────────
    from novelty_detector.dashboard.routes import bp  # noqa: PLC0415
    app.register_blueprint(bp)

    # ── Jinja2 custom filters ─────────────────────────────────────────────────
    _register_filters(app)

    logger.info("Flask app created. Debug={}", app.config["DEBUG"])
    return app


def _register_filters(app: Flask) -> None:
    """Register helper filters available in all Jinja2 templates."""

    @app.template_filter("pct")
    def _pct(value: float | None) -> str:
        """Format a 0–1 float as a percentage string, e.g. 0.923 → '92.3%'."""
        if value is None:
            return "N/A"
        return f"{value * 100:.1f}%"

    @app.template_filter("score_color")
    def _score_color(value: float | None) -> str:
        """
        Return a CSS class name based on similarity score magnitude.
        Used to colour-code score badges in the UI.
        """
        if value is None:
            return "badge-secondary"
        if value >= 0.9:
            return "badge-danger"
        if value >= 0.7:
            return "badge-warning"
        return "badge-success"

    @app.template_filter("verdict_color")
    def _verdict_color(verdict: str) -> str:
        """Return a CSS class for a verdict string."""
        return {
            "duplicate": "badge-danger",
            "near-duplicate": "badge-warning",
            "novel": "badge-success",
        }.get(verdict, "badge-secondary")
