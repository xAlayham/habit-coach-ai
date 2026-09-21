# habit-coach-ai

An AI coaching service that sits on top of a live habit-tracking product. It reads
the user's real habit data through their existing API, grounds its advice in a
curated library of habit-formation research, and streams the answer back token by
token.

## Live demo

- **App:** [habit-tracker-web-alpha.vercel.app](https://habit-tracker-web-alpha.vercel.app) — register, add a
  few habits, then open **Ask coach** in the bottom-right corner
- **API docs:** [habit-coach-ai-dnj2.onrender.com/docs](https://habit-coach-ai-dnj2.onrender.com/docs) —
  interactive Swagger UI for this service

Both backends run on Render's free tier and sleep after 15 minutes idle, so the
first request can take up to a minute while they wake. The demo database is
reset whenever the habit API restarts, so an account may need re-registering.

## The three services

| Service | Role | Stack |
| --- | --- | --- |
| [habit-tracker-api](https://github.com/xAlayham/habit-tracker-api) | REST API, JWT auth, per-user CRUD | FastAPI, on Render |
| [habit-tracker-web](https://github.com/xAlayham/habit-tracker-web) | Frontend, including the coach chat widget | React/TS, on Vercel |
| **habit-coach-ai** | **AI coaching layer (this repo)** | **FastAPI + Anthropic SDK + Chroma** |

## Architecture

```
                    +--------------------------+
                    |   habit-tracker-web      |
                    |   React/TS on Vercel     |
                    +------------+-------------+
                                 |  Authorization: Bearer <jwt>
                 +---------------+---------------+
                 v                               v
   +--------------------------+    +----------------------------------+
   |   habit-tracker-api      |    |        habit-coach-ai            |
   |   FastAPI on Render      |    |        FastAPI on Render         |
   |                          |    |                                  |
   |   POST /users/login      |    |  get_token  --> rate limiter     |
   |   GET  /habits           |    |                 (token bucket)   |
   |   GET  /habits/{id}      |    |                      |           |
   +------------^-------------+    |                      v           |
                |                  |               LRU cache (TTL)    |
                |                  |                      |           |
                |  same JWT,       |                      v           |
                |  passed through  |            +------------------+  |
                +------------------+------------+   agent loop     |  |
                                   |            |  (async, MAX 8)  |  |
                                   |            +----+--------+----+  |
                                   |                 |        |       |
                                   |                 v        v       |
                                   |         Claude Opus 5   Chroma   |
                                   |          (streaming)    vectors  |
                                   +----------------------------------+

  POST /coach          JSON answer
  POST /coach/stream   Server-Sent Events: tool_start | tool_end | text | done
  GET  /health         liveness
  GET  /stats          cache + rate-limiter counters
```

The user's JWT is never decoded for authorization here. It is bound into the tool
functions with `functools.partial` before the model sees them, so a credential can
never appear in a tool schema, a `tool_use` input, or a `tool_result`.

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

| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | - | Required |
| `HABIT_API_BASE` | - | Required, the habit-tracker-api base URL |
| `HABIT_API_TIMEOUT` | `60` | Read timeout, sized for Render cold starts |
| `CORS_ORIGINS` | localhost:3000, localhost:5173 | Comma-separated allowed origins |
| `RATE_LIMIT_BURST` | `5` | Requests allowed back-to-back |
| `RATE_LIMIT_PER_MINUTE` | `5` | Sustained refill rate |
| `CACHE_CAPACITY` | `128` | Cached answers held before LRU eviction |
| `CACHE_TTL_SECONDS` | `300` | How long a cached answer stays fresh |

## Running

```bash
uvicorn service:app --reload          # API at /coach, docs at /docs
python agent.py "how am I doing?"     # same agent from the command line
python -m pytest                      # 143 tests
```

## Docker

```bash
docker build -t habit-coach-ai .
docker run --rm -p 8000:8000 --env-file .env habit-coach-ai
```

The build is multi-stage. The builder stage installs dependencies and runs
`ingest.py`, so the embedding model and the finished vector store are baked into
the image - without that, the first request after a cold start would block for
~20 seconds downloading the model. The runtime stage copies only the virtualenv,
the model cache, the vector store and the named source files, and runs as a
non-root user. Secrets are injected at runtime and never enter the image.

## Deployment

`render.yaml` is a Render Blueprint. `ANTHROPIC_API_KEY` and `CORS_ORIGINS` are
marked `sync: false`, so Render prompts for them instead of reading them from the
repo.

Note for SSE behind a proxy: `/coach/stream` sets `X-Accel-Buffering: no`, without
which an nginx-style proxy buffers the whole response and delivers it in one lump,
silently defeating streaming.

## Grounding eval

```bash
python eval_grounding.py                    # all 20 cases (~$1.30, calls the real API)
python eval_grounding.py --only trap        # one category
python eval_grounding.py --no-judge         # deterministic checks only
python eval_grounding.py --regrade evals/results/<file>.jsonl   # re-score, no API calls
```

Checks that coaching advice uses retrieved research rather than inventing it.
Five deterministic checks plus a `claude-sonnet-5` judge (deliberately not the
model under test). The habit API is stubbed with fixtures; retrieval runs for
real. Baseline on 20 cases:

| Check | Score |
| --- | --- |
| Called research when the case needed it | 20/20 |
| Produced a citation when the case needed one | 20/20 |
| Cited nothing outside the corpus | 20/20 |
| Every citation was actually retrieved that run | 20/20 |
| Avoided the forbidden claim patterns | 20/20 |
| Judge found no unsupported claim | 11/20 |

The judge failures are consistently over-specification - the model adds concrete
numbers and examples the retrieved passages do not contain. No fabricated
citations were found.

## Notable implementation details

- **Rate limiter and LRU cache are written from scratch**, not library calls.
  The limiter is a token bucket with lazy refill, keyed on the JWT's `sub` claim
  so a re-issued token does not reset someone's quota, and it evicts buckets that
  have fully refilled. The cache is a dict plus a doubly linked list with sentinel
  nodes, O(1) get/put, capacity eviction and a TTL.
- **Cache keys include the user**, because an answer contains that user's habit
  data. Keying on the question alone would leak one user's data to another.
- **Tool results are returned in a single user message** when the model makes
  parallel calls; splitting them teaches the model to stop calling in parallel.
- **Turn exhaustion is a 504 on `/coach`** but an `error` event on `/coach/stream`,
  because once the first byte of a stream is sent the status code is already 200.

## License

MIT - see [LICENSE](LICENSE).
