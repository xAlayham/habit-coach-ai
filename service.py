import base64
import hashlib
import json
import math
import os
from typing import NamedTuple

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from agent import MODEL, AgentRefused, TurnLimitExceeded, run_agent, stream_agent
from lru import LRUCache
from ratelimit import TokenBucketRateLimiter

app = FastAPI(
    title="habit-coach-ai",
    version="0.4.0",
    description="AI coaching layer over the habit-tracker API.",
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=[
        "X-Cache",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    ],
)

RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "5"))
RATE_LIMIT_PER_MINUTE = float(os.environ.get("RATE_LIMIT_PER_MINUTE", "5"))
CACHE_CAPACITY = int(os.environ.get("CACHE_CAPACITY", "128"))
CACHE_TTL_SECONDS = float(os.environ.get("CACHE_TTL_SECONDS", "300"))

limiter = TokenBucketRateLimiter(
    capacity=RATE_LIMIT_BURST,
    refill_per_second=RATE_LIMIT_PER_MINUTE / 60.0,
)

answer_cache = LRUCache(capacity=CACHE_CAPACITY, ttl_seconds=CACHE_TTL_SECONDS)

bearer_scheme = HTTPBearer(auto_error=False)


def get_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing 'Authorization: Bearer <token>' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def rate_limit_key(token: str) -> str:
    subject = None
    parts = token.split(".")
    if len(parts) == 3:
        try:
            padded = parts[1] + "=" * (-len(parts[1]) % 4)
            subject = json.loads(base64.urlsafe_b64decode(padded)).get("sub")
        except Exception:
            subject = None
    return hashlib.sha256(str(subject or token).encode()).hexdigest()


def cache_key(token: str, question: str) -> str:
    parts = [rate_limit_key(token), MODEL, " ".join(question.split())]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


class RateLimitGrant(NamedTuple):
    token: str
    headers: dict


def enforce_rate_limit(token: str = Depends(get_token)) -> RateLimitGrant:
    decision = limiter.acquire(rate_limit_key(token))
    headers = {
        "X-RateLimit-Limit": str(decision.limit),
        "X-RateLimit-Remaining": str(decision.remaining),
        "X-RateLimit-Reset": str(math.ceil(decision.reset_after)),
    }

    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Slow down and retry shortly.",
            headers={
                **headers,
                "Retry-After": str(max(1, math.ceil(decision.retry_after))),
            },
        )

    return RateLimitGrant(token, headers)


class CoachRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class CoachResponse(BaseModel):
    answer: str


def sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    snapshot = answer_cache.stats()
    return {
        "cache": {
            "hits": snapshot.hits,
            "misses": snapshot.misses,
            "evictions": snapshot.evictions,
            "expirations": snapshot.expirations,
            "size": snapshot.size,
            "capacity": snapshot.capacity,
            "hit_rate": round(snapshot.hit_rate, 3),
        },
        "rate_limit": {
            "burst": limiter.capacity,
            "per_minute": round(limiter.refill_per_second * 60, 2),
            "tracked_users": len(limiter),
        },
    }


@app.post("/coach", response_model=CoachResponse)
async def coach(
    payload: CoachRequest,
    response: Response,
    grant: RateLimitGrant = Depends(enforce_rate_limit),
) -> CoachResponse:
    key = cache_key(grant.token, payload.question)
    cached = answer_cache.get(key)

    if cached is not None:
        response.headers.update({**grant.headers, "X-Cache": "HIT"})
        return CoachResponse(answer=cached)

    response.headers.update({**grant.headers, "X-Cache": "MISS"})

    try:
        answer = await run_agent(payload.question, grant.token)
    except TurnLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
            headers=grant.headers,
        ) from exc
    except AgentRefused as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
            headers=grant.headers,
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not reach the Anthropic API.",
            headers=grant.headers,
        ) from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Anthropic API error ({exc.status_code}).",
            headers=grant.headers,
        ) from exc

    answer_cache.put(key, answer)
    return CoachResponse(answer=answer)


@app.post("/coach/stream")
async def coach_stream(
    payload: CoachRequest,
    grant: RateLimitGrant = Depends(enforce_rate_limit),
):
    key = cache_key(grant.token, payload.question)
    cached = answer_cache.get(key)

    async def replay_cached():
        yield sse({"type": "text", "delta": cached})
        yield sse({"type": "done", "answer": cached, "cached": True})

    async def event_source():
        answer = None
        try:
            async for event in stream_agent(payload.question, grant.token):
                if event["type"] == "done":
                    answer = event["answer"]
                yield sse(event)
        except anthropic.APIConnectionError:
            yield sse({
                "type": "error",
                "code": "upstream",
                "detail": "Could not reach the Anthropic API.",
            })
        except anthropic.APIStatusError as exc:
            yield sse({
                "type": "error",
                "code": "upstream",
                "detail": f"Anthropic API error ({exc.status_code}).",
            })

        if answer is not None:
            answer_cache.put(key, answer)

    return StreamingResponse(
        replay_cached() if cached is not None else event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Cache": "HIT" if cached is not None else "MISS",
            **grant.headers,
        },
    )
