import anthropic
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from agent import AgentRefused, TurnLimitExceeded, run_agent

app = FastAPI(
    title="habit-coach-ai",
    version="0.1.0",
    description="AI coaching layer over the habit-tracker API.",
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


class CoachRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class CoachResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/coach", response_model=CoachResponse)
def coach(payload: CoachRequest, token: str = Depends(get_token)) -> CoachResponse:
    try:
        answer = run_agent(payload.question, token)
    except TurnLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)
        ) from exc
    except AgentRefused as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not reach the Anthropic API.",
        ) from exc
    except anthropic.APIStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Anthropic API error ({exc.status_code}).",
        ) from exc

    return CoachResponse(answer=answer)
