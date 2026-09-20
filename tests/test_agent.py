import asyncio

import pytest

import agent
from fakes import calls_tools, collect, refuses, says

pytestmark = pytest.mark.anyio

HABIT_READ = {
    "id": 7,
    "name": "Read 40 pages",
    "frequency": "daily",
    "completed": False,
    "streak_count": 3,
    "last_completed_date": "2026-09-19",
}
HABIT_RUN = {
    "id": 9,
    "name": "Run",
    "frequency": "weekly",
    "completed": True,
    "streak_count": 1,
    "last_completed_date": "2026-09-20",
}


def tool_results_of(messages, index):
    return messages[index]["content"]


def event_types(events):
    return [event["type"] for event in events]


def streamed_text(events):
    return "".join(event["delta"] for event in events if event["type"] == "text")


async def test_single_tool_call(habit_api, use_model):
    habit_api.route("/habits", json=[HABIT_READ])
    model = use_model(
        calls_tools(("get_user_habits", {})),
        says("You are tracking one habit: Read 40 pages."),
    )

    answer = await agent.run_agent("what am I tracking?", "jwt-abc")

    assert answer == "You are tracking one habit: Read 40 pages."
    assert len(model.messages.calls) == 2
    assert habit_api.paths == ["/habits"]
    assert habit_api.auth_headers == ["Bearer jwt-abc"]

    results = tool_results_of(model.sent_messages(1), -1)
    assert len(results) == 1
    assert results[0]["tool_use_id"] == "toolu_0"
    assert results[0]["is_error"] is False
    assert "Read 40 pages" in results[0]["content"]


async def test_multi_step_chaining(habit_api, use_model):
    habit_api.route("/habits", json=[HABIT_READ])
    habit_api.route("/habits/7", json=HABIT_READ)
    model = use_model(
        calls_tools(("get_user_habits", {})),
        calls_tools(("get_habit_detail", {"habit_id": 7})),
        says("Your reading streak is 3 days."),
    )

    answer = await agent.run_agent("how is my reading habit?", "jwt-abc")

    assert answer == "Your reading streak is 3 days."
    assert len(model.messages.calls) == 3
    assert habit_api.paths == ["/habits", "/habits/7"]

    second_result = tool_results_of(model.sent_messages(2), -1)[0]
    assert "streak_count" in second_result["content"]


async def test_parallel_tool_calls(habit_api, use_model):
    habit_api.route("/habits/7", json=HABIT_READ)
    habit_api.route("/habits/9", json=HABIT_RUN)
    model = use_model(
        calls_tools(
            ("get_habit_detail", {"habit_id": 7}),
            ("get_habit_detail", {"habit_id": 9}),
        ),
        says("Reading is ahead of running."),
    )

    answer = await agent.run_agent("compare reading and running", "jwt-abc")

    assert answer == "Reading is ahead of running."
    assert sorted(habit_api.paths) == ["/habits/7", "/habits/9"]

    follow_up = model.sent_messages(1)
    assert follow_up[-1]["role"] == "user"
    results = follow_up[-1]["content"]
    assert len(results) == 2
    assert [r["tool_use_id"] for r in results] == ["toolu_0", "toolu_1"]


async def test_parallel_tools_actually_run_concurrently(habit_api, use_model, monkeypatch):
    barrier = asyncio.Barrier(2)

    async def blocks_until_both_started(token, habit_id):
        await barrier.wait()
        return {"id": habit_id}

    monkeypatch.setattr(agent, "get_habit_detail", blocks_until_both_started)
    use_model(
        calls_tools(
            ("get_habit_detail", {"habit_id": 7}),
            ("get_habit_detail", {"habit_id": 9}),
        ),
        says("Both done."),
    )

    async with asyncio.timeout(3):
        answer = await agent.run_agent("compare", "jwt-abc")

    assert answer == "Both done."


async def test_tool_http_error_is_reported_not_raised(habit_api, use_model):
    model = use_model(
        calls_tools(("get_habit_detail", {"habit_id": 99})),
        says("I could not find a habit with that id."),
    )

    answer = await agent.run_agent("how is habit 99?", "jwt-abc")

    assert answer == "I could not find a habit with that id."
    result = tool_results_of(model.sent_messages(1), -1)[0]
    assert result["is_error"] is True
    assert "404" in result["content"]


async def test_unknown_tool_name_is_reported_not_raised(habit_api, use_model):
    model = use_model(
        calls_tools(("get_habit_history", {})),
        says("That tool is not available."),
    )

    answer = await agent.run_agent("show my history", "jwt-abc")

    assert answer == "That tool is not available."
    result = tool_results_of(model.sent_messages(1), -1)[0]
    assert result["is_error"] is True
    assert "Tool call failed" in result["content"]


async def test_turn_exhaustion(habit_api, use_model):
    habit_api.route("/habits", json=[HABIT_READ])
    model = use_model(calls_tools(("get_user_habits", {})), repeat_last=True)

    with pytest.raises(agent.TurnLimitExceeded):
        await agent.run_agent("loop forever", "jwt-abc")

    assert len(model.messages.calls) == agent.MAX_TURNS


async def test_refusal_raises(habit_api, use_model):
    use_model(refuses("Declined for policy reasons."))

    with pytest.raises(agent.AgentRefused, match="Declined for policy reasons."):
        await agent.run_agent("something disallowed", "jwt-abc")


async def test_token_never_reaches_the_model(habit_api, use_model):
    habit_api.route("/habits", json=[HABIT_READ])
    model = use_model(
        calls_tools(("get_user_habits", {})),
        says("All good."),
    )

    await agent.run_agent("what am I tracking?", "super-secret-jwt")

    assert "super-secret-jwt" not in str(model.messages.calls)
    assert "super-secret-jwt" not in str(agent.TOOLS)
    assert habit_api.auth_headers == ["Bearer super-secret-jwt"]


async def test_conversations_do_not_share_state(habit_api, use_model):
    habit_api.route("/habits", json=[HABIT_READ])
    model = use_model(
        calls_tools(("get_user_habits", {})),
        says("First answer."),
        calls_tools(("get_user_habits", {})),
        says("Second answer."),
    )

    first = await agent.run_agent("question one", "jwt-one")
    second = await agent.run_agent("question two", "jwt-two")

    assert (first, second) == ("First answer.", "Second answer.")
    assert model.sent_messages(2)[0]["content"] == "question two"
    assert len(model.sent_messages(2)) == 1
    assert habit_api.auth_headers == ["Bearer jwt-one", "Bearer jwt-two"]


async def test_stream_emits_text_deltas_then_done(habit_api, use_model):
    use_model(says("Keep your streak going."))

    events = await collect(agent.stream_agent("how am I doing?", "jwt-abc"))

    assert event_types(events)[-1] == "done"
    assert set(event_types(events)) == {"text", "done"}
    assert streamed_text(events) == "Keep your streak going."
    assert events[-1]["answer"] == "Keep your streak going."
    assert len([event for event in events if event["type"] == "text"]) > 1


async def test_stream_emits_preamble_before_tool_calls(habit_api, use_model):
    habit_api.route("/habits", json=[HABIT_READ])
    use_model(
        calls_tools(("get_user_habits", {}), preamble="Let me check your habits."),
        says("You have one habit."),
    )

    events = await collect(agent.stream_agent("what am I tracking?", "jwt-abc"))
    types = event_types(events)

    assert types[0] == "text"
    assert types.index("text") < types.index("tool_start") < types.index("tool_end")
    assert types[-1] == "done"
    assert streamed_text(events) == "Let me check your habits.You have one habit."


async def test_stream_reports_tool_lifecycle(habit_api, use_model):
    habit_api.route("/habits", json=[HABIT_READ])
    use_model(
        calls_tools(("get_user_habits", {})),
        says("Done."),
    )

    events = await collect(agent.stream_agent("what am I tracking?", "jwt-abc"))
    starts = [event for event in events if event["type"] == "tool_start"]
    ends = [event for event in events if event["type"] == "tool_end"]

    assert [event["name"] for event in starts] == ["get_user_habits"]
    assert starts[0]["input"] == {}
    assert ends[0] == {
        "type": "tool_end",
        "name": "get_user_habits",
        "ok": True,
        "detail": None,
    }


async def test_stream_reports_a_failing_tool(habit_api, use_model):
    use_model(
        calls_tools(("get_habit_detail", {"habit_id": 99})),
        says("Could not find it."),
    )

    events = await collect(agent.stream_agent("habit 99?", "jwt-abc"))
    end = next(event for event in events if event["type"] == "tool_end")

    assert end["ok"] is False
    assert "404" in end["detail"]


async def test_stream_ends_with_an_error_event_on_turn_limit(habit_api, use_model):
    habit_api.route("/habits", json=[HABIT_READ])
    use_model(calls_tools(("get_user_habits", {})), repeat_last=True)

    events = await collect(agent.stream_agent("loop forever", "jwt-abc"))

    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "turn_limit"
    assert "done" not in event_types(events)


async def test_stream_ends_with_an_error_event_on_refusal(habit_api, use_model):
    use_model(refuses("Declined."))

    events = await collect(agent.stream_agent("disallowed", "jwt-abc"))

    assert events[-1] == {"type": "error", "code": "refused", "detail": "Declined."}
