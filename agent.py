import functools
import json
import os
import sys

import anthropic
import httpx
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-5"
MAX_TURNS = 8

client = anthropic.Anthropic()
http_client = httpx.Client(
    base_url=os.environ["HABIT_API_BASE"].rstrip("/"),
    timeout=30.0,
)

SYSTEM_PROMPT = (
    "You are a habit coach. Use the available tools to look up the user's real "
    "habit data before giving advice. Never invent numbers or guess at progress - "
    "if you need data, call a tool. "
    "You can see each habit's current streak, whether it is completed for the "
    "current period, and the date it was last completed. You CANNOT see a full "
    "history of past completions, so do not claim to know how many days someone "
    "missed or how they did in a specific past week. "
    "Keep advice concise, specific, and grounded in what the data actually shows."
)


class AgentError(Exception):
    pass


class TurnLimitExceeded(AgentError):
    pass


class AgentRefused(AgentError):
    pass


def _api_get(path: str, token: str):
    response = http_client.get(path, headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return response.json()


def get_user_habits(token: str) -> list[dict]:
    habits = _api_get("/habits", token)
    return [
        {"id": h["id"], "name": h["name"], "frequency": h["frequency"]}
        for h in habits
    ]


def get_habit_detail(token: str, habit_id: int) -> dict:
    return _api_get(f"/habits/{habit_id}", token)


get_user_habits_tool = {
    "name": "get_user_habits",
    "description": (
        "Lists every habit the user is tracking, with its numeric id, name and "
        "frequency (how often it is meant to be done). Takes no arguments. Call "
        "this first whenever you need to know what habits exist, or to find the "
        "id of a habit the user mentioned by name."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

get_habit_detail_tool = {
    "name": "get_habit_detail",
    "description": (
        "Gets the current progress for ONE habit: its streak_count (consecutive "
        "periods completed), whether it is completed for the current period, and "
        "last_completed_date. Requires the numeric habit_id - call get_user_habits "
        "first to look up the id for the habit the user means. To compare several "
        "habits, call this once per habit."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "habit_id": {
                "type": "integer",
                "description": "The numeric id from get_user_habits, not the habit's name.",
            },
        },
        "required": ["habit_id"],
    },
}

TOOLS = [get_user_habits_tool, get_habit_detail_tool]


def build_tool_functions(token: str) -> dict:
    return {
        "get_user_habits": functools.partial(get_user_habits, token),
        "get_habit_detail": functools.partial(get_habit_detail, token),
    }


def _run_tool_calls(content, tool_functions: dict, verbose: bool) -> list[dict]:
    tool_results = []

    for block in content:
        if block.type != "tool_use":
            continue

        if verbose:
            print(f"   calling {block.name} with {block.input}")

        try:
            fn = tool_functions[block.name]
            result = fn(**block.input)
            result_content = json.dumps(result)
            is_error = False
        except httpx.HTTPStatusError as exc:
            result_content = (
                f"API returned {exc.response.status_code}: {exc.response.text}"
            )
            is_error = True
        except Exception as exc:
            result_content = f"Tool call failed: {exc}"
            is_error = True

        if is_error and verbose:
            print(f"   -> {result_content}")

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result_content,
            "is_error": is_error,
        })

    return tool_results


def _final_text(response) -> str:
    return "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def run_agent(question: str, token: str, verbose: bool = False) -> str:
    tool_functions = build_tool_functions(token)
    messages = [{"role": "user", "content": question}]

    for turn in range(MAX_TURNS):
        if verbose:
            print(f"--- turn {turn + 1} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "explanation", None)
            raise AgentRefused(detail or "The model declined to answer this request.")

        if response.stop_reason != "tool_use":
            return _final_text(response)

        messages.append({
            "role": "user",
            "content": _run_tool_calls(response.content, tool_functions, verbose),
        })

    raise TurnLimitExceeded(
        f"The agent did not reach a final answer within {MAX_TURNS} turns."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: provide a question. Usage: python agent.py 'your question'")
        sys.exit(1)

    try:
        print(run_agent(sys.argv[1], os.environ["HABIT_API_TOKEN"], verbose=True))
    except AgentError as exc:
        print(f"\n[error] {exc}")
        sys.exit(1)
