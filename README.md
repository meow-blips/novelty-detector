# Novelty & Duplicate Detection System

A staged, two-phase pipeline for detecting exact copies and paraphrased
near-duplicates in text document corpora.

## Architecture

```
novelty_detector/
├── ingestion/      — Cleaning, tokenisation, normalisation
├── lexical/        — MinHash + LSH candidate retrieval (Stage 1)
├── semantic/       — Sentence-embedding cosine similarity (Stage 2)
├── classification/ — Thresholded labelling + audit logging
├── storage/        — SQLAlchemy models & DB session management
└── dashboard/      — Flask reviewer UI (Phase 3)
```

## Quick Start

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt punkt_tab
python -m spacy download en_core_web_sm
python -c "from novelty_detector.lexical.lsh_index import LSHIndex; print('OK')"
```
