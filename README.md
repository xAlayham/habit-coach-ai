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

Optional rate-limit tuning (per user, applied to `/coach` and `/coach/stream`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `RATE_LIMIT_BURST` | `5` | Requests allowed back-to-back |
| `RATE_LIMIT_PER_MINUTE` | `5` | Sustained refill rate |
| `CACHE_CAPACITY` | `128` | Cached answers held before LRU eviction |
| `CACHE_TTL_SECONDS` | `300` | How long a cached answer stays fresh |

## Running

```bash
uvicorn service:app --reload          # API at /coach, docs at /docs
python agent.py "how am I doing?"     # same agent from the command line
python -m pytest                      # test suite
```

`GET /stats` reports live cache and rate-limiter counters.

## Grounding eval

```bash
python eval_grounding.py                    # all 20 cases (~$1.30, calls the real API)
python eval_grounding.py --only trap        # one category
python eval_grounding.py --no-judge         # deterministic checks only
python eval_grounding.py --regrade evals/results/<file>.jsonl   # re-score, no API calls
```

Checks that coaching advice uses retrieved research rather than inventing it.
Five deterministic checks plus a `claude-sonnet-5` judge. The habit API is
stubbed with fixtures; retrieval runs for real. Baseline on 20 cases:

| Check | Score |
| --- | --- |
| Called research when the case needed it | 20/20 |
| Produced a citation when the case needed one | 20/20 |
| Cited nothing outside the corpus | 20/20 |
| Every citation was actually retrieved that run | 20/20 |
| Avoided the forbidden claim patterns | 20/20 |
| Judge found no unsupported claim | 11/20 |
