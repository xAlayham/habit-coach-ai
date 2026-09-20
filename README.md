# habit-coach-ai

An intelligent companion for your Habit Tracker app that analyzes your routine data, delivers personalized coaching based on proven habits, and responds instantly.

## Setup

```bash
pip install -r requirements-dev.txt   # or requirements.txt for production only
python ingest.py --rebuild            # builds the research vector store
```

`ingest.py` is required before the coach can ground its advice: the vector store
lives in `chroma_db/`, which is not committed. The first run downloads the
embedding model (~80MB) to your user cache.

Create a `.env` with `ANTHROPIC_API_KEY`, `HABIT_API_BASE` and, for the CLI only,
`HABIT_API_TOKEN`.

## Running

```bash
uvicorn service:app --reload          # API at /coach, docs at /docs
python agent.py "how am I doing?"     # same agent from the command line
python -m pytest                      # test suite
```
