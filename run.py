"""
run.py
=======
Development server entry point.

Usage
-----
    python run.py

Production
----------
Use a production WSGI server instead:

    # Gunicorn (Linux/macOS)
    gunicorn "novelty_detector.dashboard.app:create_app()" --bind 0.0.0.0:5000

    # Waitress (Windows-friendly)
    pip install waitress
    waitress-serve --port=5000 "novelty_detector.dashboard.app:create_app()"
"""

from novelty_detector.dashboard.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=app.config.get("DEBUG", False),
        use_reloader=True,
    )
