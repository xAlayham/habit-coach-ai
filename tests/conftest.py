import os

os.environ["HABIT_API_BASE"] = "https://habit-api.test"
os.environ["ANTHROPIC_API_KEY"] = "test-key-not-real"

import pytest

import agent
from fakes import FakeAnthropic, FakeHabitAPI

HABIT_API_BASE = os.environ["HABIT_API_BASE"]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def habit_api(monkeypatch):
    fake = FakeHabitAPI(HABIT_API_BASE)
    monkeypatch.setattr(agent, "http_client", fake.build_client())
    return fake


@pytest.fixture
def use_model(monkeypatch):
    def install(*responses, repeat_last=False):
        fake = FakeAnthropic(*responses, repeat_last=repeat_last)
        monkeypatch.setattr(agent, "client", fake)
        return fake

    return install
