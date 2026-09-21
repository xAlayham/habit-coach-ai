import base64
import hashlib
import json
import math
import os
from typing import NamedTuple

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from agent import AgentRefused, TurnLimitExceeded, run_agent, stream_agent
from ratelimit import TokenBucketRateLimiter

app = FastAPI(
    title="habit-coach-ai",
    version="0.3.0",
    description="AI coaching layer over the habit-tracker API.",
)

RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "5"))
RATE_LIMIT_PER_MINUTE = float(os.environ.get("RATE_LIMIT_PER_MINUTE", "5"))

limiter = TokenBucketRateLimiter(
    capacity=RATE_LIMIT_BURST,
    refill_per_second=RATE_LIMIT_PER_MINUTE / 60.0,
)

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


@app.post("/coach", response_model=CoachResponse)
async def coach(
    payload: CoachRequest,
    response: Response,
    grant: RateLimitGrant = Depends(enforce_rate_limit),
) -> CoachResponse:
    response.headers.update(grant.headers)

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

    return CoachResponse(answer=answer)


@app.post("/coach/stream")
async def coach_stream(
    payload: CoachRequest,
    grant: RateLimitGrant = Depends(enforce_rate_limit),
):
    async def event_source():
        try:
            async for event in stream_agent(payload.question, grant.token):
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

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            **grant.headers,
        },
    )
