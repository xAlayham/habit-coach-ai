import json

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

import service
from agent import AgentRefused, TurnLimitExceeded

OVERRIDE_TOKEN = "override-jwt"


@pytest.fixture
def client():
    service.app.dependency_overrides[service.get_token] = lambda: OVERRIDE_TOKEN
    yield TestClient(service.app)
    service.app.dependency_overrides.clear()


@pytest.fixture
def unauthenticated_client():
    return TestClient(service.app)


@pytest.fixture
def agent_returns(monkeypatch):
    def install(answer=None, raises=None):
        received = {}

        async def fake_run_agent(question, token):
            received["question"] = question
            received["token"] = token
            if raises is not None:
                raise raises
            return answer

        monkeypatch.setattr(service, "run_agent", fake_run_agent)
        return received

    return install


@pytest.fixture
def agent_streams(monkeypatch):
    def install(events=(), raises=None):
        received = {}

        async def fake_stream_agent(question, token):
            received["question"] = question
            received["token"] = token
            for event in events:
                yield event
            if raises is not None:
                raise raises

        monkeypatch.setattr(service, "stream_agent", fake_stream_agent)
        return received

    return install


def parse_sse(body):
    parsed = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        parsed.append((lines["event"], json.loads(lines["data"])))
    return parsed


def test_health_needs_no_auth(unauthenticated_client):
    response = unauthenticated_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_authorization_header_is_401(unauthenticated_client):
    response = unauthenticated_client.post("/coach", json={"question": "hi"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_real_dependency_extracts_the_bearer_token(unauthenticated_client, agent_returns):
    received = agent_returns(answer="ok")

    response = unauthenticated_client.post(
        "/coach",
        json={"question": "hi"},
        headers={"Authorization": "Bearer real-jwt-from-header"},
    )

    assert response.status_code == 200
    assert received["token"] == "real-jwt-from-header"


def test_coach_returns_the_agent_answer(client, agent_returns):
    received = agent_returns(answer="Keep your streak going.")

    response = client.post("/coach", json={"question": "how am I doing?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Keep your streak going."}
    assert received == {"question": "how am I doing?", "token": OVERRIDE_TOKEN}


def test_empty_question_is_rejected_before_the_model(client, agent_returns):
    received = agent_returns(answer="should not be reached")

    response = client.post("/coach", json={"question": ""})

    assert response.status_code == 422
    assert received == {}


def test_turn_exhaustion_is_504(client, agent_returns):
    agent_returns(raises=TurnLimitExceeded("out of turns"))

    response = client.post("/coach", json={"question": "hi"})

    assert response.status_code == 504
    assert response.json()["detail"] == "out of turns"


def test_refusal_is_422(client, agent_returns):
    agent_returns(raises=AgentRefused("declined"))

    response = client.post("/coach", json={"question": "hi"})

    assert response.status_code == 422
    assert response.json()["detail"] == "declined"


def test_anthropic_connection_failure_is_503(client, agent_returns):
    agent_returns(
        raises=anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.test"))
    )

    response = client.post("/coach", json={"question": "hi"})

    assert response.status_code == 503


def test_anthropic_status_error_is_502(client, agent_returns):
    agent_returns(
        raises=anthropic.APIStatusError(
            "boom",
            response=httpx.Response(
                500, request=httpx.Request("POST", "https://api.test")
            ),
            body=None,
        )
    )

    response = client.post("/coach", json={"question": "hi"})

    assert response.status_code == 502


def test_stream_requires_auth(unauthenticated_client):
    response = unauthenticated_client.post("/coach/stream", json={"question": "hi"})

    assert response.status_code == 401


def test_stream_sets_event_stream_headers(client, agent_streams):
    agent_streams(events=[{"type": "done", "answer": "ok"}])

    response = client.post("/coach/stream", json={"question": "hi"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"


def test_stream_emits_one_sse_frame_per_event(client, agent_streams):
    received = agent_streams(events=[
        {"type": "text", "delta": "Keep "},
        {"type": "tool_start", "name": "get_user_habits", "input": {}},
        {"type": "tool_end", "name": "get_user_habits", "ok": True, "detail": None},
        {"type": "text", "delta": "going."},
        {"type": "done", "answer": "Keep going."},
    ])

    response = client.post("/coach/stream", json={"question": "how am I doing?"})
    frames = parse_sse(response.text)

    assert [name for name, _ in frames] == [
        "text",
        "tool_start",
        "tool_end",
        "text",
        "done",
    ]
    assert frames[0][1] == {"type": "text", "delta": "Keep "}
    assert frames[-1][1]["answer"] == "Keep going."
    assert received == {"question": "how am I doing?", "token": OVERRIDE_TOKEN}


def test_stream_reports_turn_limit_as_an_event_not_a_status_code(client, agent_streams):
    agent_streams(events=[
        {"type": "error", "code": "turn_limit", "detail": "out of turns"},
    ])

    response = client.post("/coach/stream", json={"question": "hi"})
    frames = parse_sse(response.text)

    assert response.status_code == 200
    assert frames == [("error", {"type": "error", "code": "turn_limit", "detail": "out of turns"})]


def test_stream_converts_a_mid_stream_api_failure_into_an_error_event(client, agent_streams):
    agent_streams(
        events=[{"type": "text", "delta": "partial"}],
        raises=anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.test")),
    )

    response = client.post("/coach/stream", json={"question": "hi"})
    frames = parse_sse(response.text)

    assert response.status_code == 200
    assert [name for name, _ in frames] == ["text", "error"]
    assert frames[-1][1]["code"] == "upstream"


def test_stream_rejects_an_empty_question(client, agent_streams):
    received = agent_streams(events=[{"type": "done", "answer": "x"}])

    response = client.post("/coach/stream", json={"question": ""})

    assert response.status_code == 422
    assert received == {}
