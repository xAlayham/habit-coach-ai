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

        def fake_run_agent(question, token):
            received["question"] = question
            received["token"] = token
            if raises is not None:
                raise raises
            return answer

        monkeypatch.setattr(service, "run_agent", fake_run_agent)
        return received

    return install


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
