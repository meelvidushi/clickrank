# ClickRank

A search ranking system that combines TF-IDF, PageRank, and query history for personalized results.

## Data
The system learns from user behavior by storing query history in `queries.json` and using semantic embeddings to understand which pages users find relevant for similar queries.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The script will auto-download required NLTK data on first run

## Requirements

Create `requirements.txt`:
```
beautifulsoup4
numpy
nltk
scikit-learn
sentence-transformers
networkx
```

## Usage

Run a search query:
```bash
python search_engine.py "repairable car"
```

Enable verbose output:
```bash
python search_engine.py "repairable car" --verbose
```

## Output

The script compares two ranking methods:
- **Normal Ranking**: TF-IDF × PageRank
- **ClickRank**: Adds personalization based on query history (stored in `queries.json`)
- **NDCG scores**: Evaluation metrics comparing both methods

Returns ranked pages and NDCG evaluation scores.
