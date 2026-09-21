import base64
import json

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

import service
from agent import AgentRefused, TurnLimitExceeded
from ratelimit import TokenBucketRateLimiter

OVERRIDE_TOKEN = "override-jwt"


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    monkeypatch.setattr(
        service,
        "limiter",
        TokenBucketRateLimiter(capacity=100, refill_per_second=100.0),
    )


@pytest.fixture
def tight_limiter(monkeypatch):
    def install(capacity=2, refill_per_second=0.5):
        limiter = TokenBucketRateLimiter(
            capacity=capacity, refill_per_second=refill_per_second
        )
        monkeypatch.setattr(service, "limiter", limiter)
        return limiter

    return install


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


def jwt_with_sub(subject, nonce="a"):
    def segment(raw):
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    header = segment(b'{"alg":"HS256","typ":"JWT"}')
    payload = segment(json.dumps({"sub": subject, "nonce": nonce}).encode())
    return f"{header}.{payload}.signature-{nonce}"


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


def test_rate_limit_key_is_stable_across_reissued_tokens():
    first = service.rate_limit_key(jwt_with_sub("user-42", nonce="monday"))
    second = service.rate_limit_key(jwt_with_sub("user-42", nonce="tuesday"))
    other = service.rate_limit_key(jwt_with_sub("user-99", nonce="monday"))

    assert first == second
    assert first != other


def test_rate_limit_key_falls_back_to_the_whole_token():
    assert service.rate_limit_key("not-a-jwt") == service.rate_limit_key("not-a-jwt")
    assert service.rate_limit_key("not-a-jwt") != service.rate_limit_key("other")


def test_rate_limit_key_does_not_leak_the_token():
    token = jwt_with_sub("user-42")

    key = service.rate_limit_key(token)

    assert token not in key
    assert "user-42" not in key


def test_successful_response_carries_rate_limit_headers(client, agent_returns, tight_limiter):
    tight_limiter(capacity=5)
    agent_returns(answer="ok")

    response = client.post("/coach", json={"question": "hi"})

    assert response.headers["x-ratelimit-limit"] == "5"
    assert response.headers["x-ratelimit-remaining"] == "4"
    assert int(response.headers["x-ratelimit-reset"]) >= 1


def test_stream_response_carries_rate_limit_headers(client, agent_streams, tight_limiter):
    tight_limiter(capacity=5)
    agent_streams(events=[{"type": "done", "answer": "ok"}])

    response = client.post("/coach/stream", json={"question": "hi"})

    assert response.headers["x-ratelimit-limit"] == "5"
    assert response.headers["x-ratelimit-remaining"] == "4"


def test_burst_then_429_with_retry_after(client, agent_returns, tight_limiter):
    tight_limiter(capacity=2, refill_per_second=0.5)
    agent_returns(answer="ok")

    first = client.post("/coach", json={"question": "hi"})
    second = client.post("/coach", json={"question": "hi"})
    third = client.post("/coach", json={"question": "hi"})

    assert (first.status_code, second.status_code) == (200, 200)
    assert third.status_code == 429
    assert third.headers["retry-after"] == "2"
    assert third.headers["x-ratelimit-remaining"] == "0"
    assert third.headers["x-ratelimit-limit"] == "2"
    assert "Rate limit exceeded" in third.json()["detail"]


def test_rate_limited_request_never_reaches_the_model(client, agent_returns, tight_limiter):
    tight_limiter(capacity=1)
    received = agent_returns(answer="ok")

    client.post("/coach", json={"question": "first"})
    received.clear()
    blocked = client.post("/coach", json={"question": "second"})

    assert blocked.status_code == 429
    assert received == {}


def test_stream_endpoint_is_rate_limited_too(client, agent_streams, tight_limiter):
    tight_limiter(capacity=1)
    agent_streams(events=[{"type": "done", "answer": "ok"}])

    assert client.post("/coach/stream", json={"question": "hi"}).status_code == 200
    assert client.post("/coach/stream", json={"question": "hi"}).status_code == 429


def test_both_endpoints_share_one_budget(client, agent_returns, agent_streams, tight_limiter):
    tight_limiter(capacity=1)
    agent_returns(answer="ok")
    agent_streams(events=[{"type": "done", "answer": "ok"}])

    assert client.post("/coach", json={"question": "hi"}).status_code == 200
    assert client.post("/coach/stream", json={"question": "hi"}).status_code == 429


def test_users_do_not_share_a_budget(unauthenticated_client, agent_returns, tight_limiter):
    tight_limiter(capacity=1)
    agent_returns(answer="ok")
    alice = {"Authorization": f"Bearer {jwt_with_sub('alice')}"}
    bob = {"Authorization": f"Bearer {jwt_with_sub('bob')}"}

    assert unauthenticated_client.post("/coach", json={"question": "hi"}, headers=alice).status_code == 200
    assert unauthenticated_client.post("/coach", json={"question": "hi"}, headers=alice).status_code == 429
    assert unauthenticated_client.post("/coach", json={"question": "hi"}, headers=bob).status_code == 200


def test_a_reissued_jwt_does_not_reset_the_budget(unauthenticated_client, agent_returns, tight_limiter):
    tight_limiter(capacity=1)
    agent_returns(answer="ok")
    monday = {"Authorization": f"Bearer {jwt_with_sub('alice', nonce='monday')}"}
    tuesday = {"Authorization": f"Bearer {jwt_with_sub('alice', nonce='tuesday')}"}

    assert unauthenticated_client.post("/coach", json={"question": "hi"}, headers=monday).status_code == 200
    assert unauthenticated_client.post("/coach", json={"question": "hi"}, headers=tuesday).status_code == 429


def test_health_is_never_rate_limited(unauthenticated_client, tight_limiter):
    tight_limiter(capacity=1)

    for _ in range(5):
        assert unauthenticated_client.get("/health").status_code == 200


def test_unauthenticated_requests_do_not_consume_anyone_s_budget(
    unauthenticated_client, agent_returns, tight_limiter
):
    tight_limiter(capacity=1)
    agent_returns(answer="ok")
    alice = {"Authorization": f"Bearer {jwt_with_sub('alice')}"}

    for _ in range(3):
        assert unauthenticated_client.post("/coach", json={"question": "hi"}).status_code == 401

    assert unauthenticated_client.post("/coach", json={"question": "hi"}, headers=alice).status_code == 200


def test_malformed_requests_still_cost_quota(client, agent_returns, tight_limiter):
    limiter = tight_limiter(capacity=2)
    agent_returns(answer="ok")

    assert client.post("/coach", json={"question": ""}).status_code == 422
    assert client.post("/coach", json={"question": ""}).status_code == 422
    assert client.post("/coach", json={"question": "valid"}).status_code == 429
    assert len(limiter) == 1
