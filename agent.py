import asyncio
import functools
import json
import os
import sys

import anthropic
import httpx
from dotenv import load_dotenv

import rag

load_dotenv()

MODEL = "claude-opus-5"
MAX_TURNS = 8

client = anthropic.AsyncAnthropic()
http_client = httpx.AsyncClient(
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
    "Keep advice concise, specific, and grounded in what the data actually shows. "
    "When the user asks anything about how habits work in general - how long they "
    "take to form, what a missed day or broken streak means, how to make one stick, "
    "why they keep forgetting - call search_habit_research first and base your "
    "advice on what it returns. Cite the source inline, author and year is enough, "
    "for example (Lally et al., 2010). Never attribute a claim to a source that did "
    "not make it, and never invent a citation. If the research library does not "
    "cover the question, say so plainly and give your best general advice without "
    "a citation."
)


class AgentError(Exception):
    pass


class TurnLimitExceeded(AgentError):
    pass


class AgentRefused(AgentError):
    pass


async def _api_get(path: str, token: str):
    response = await http_client.get(path, headers={"Authorization": f"Bearer {token}"})
    response.raise_for_status()
    return response.json()


async def get_user_habits(token: str) -> list[dict]:
    habits = await _api_get("/habits", token)
    return [
        {"id": h["id"], "name": h["name"], "frequency": h["frequency"]}
        for h in habits
    ]


async def get_habit_detail(token: str, habit_id: int) -> dict:
    return await _api_get(f"/habits/{habit_id}", token)


async def search_habit_research(query: str, top_k: int = rag.DEFAULT_TOP_K) -> list[dict]:
    return await asyncio.to_thread(rag.search_research, query, top_k)


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

search_habit_research_tool = {
    "name": "search_habit_research",
    "description": (
        "Searches a curated library of habit-formation research - peer-reviewed "
        "studies plus a few practitioner frameworks - and returns the most relevant "
        "passages, each with its citation. Call this whenever the user asks how "
        "habits work in general: how long they take to become automatic, what a "
        "missed day or broken streak actually costs, how to make a habit stick, why "
        "they keep forgetting one. It contains NO data about this user - use "
        "get_user_habits and get_habit_detail for that. Search with the user's "
        "underlying problem phrased in natural language, for example 'broke a long "
        "streak and feels like giving up', rather than with single keywords."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's problem or question in natural language.",
            },
            "top_k": {
                "type": "integer",
                "description": "How many distinct sources to return, 1 to 5. Defaults to 3.",
            },
        },
        "required": ["query"],
    },
}

TOOLS = [get_user_habits_tool, get_habit_detail_tool, search_habit_research_tool]


def build_tool_functions(token: str) -> dict:
    return {
        "get_user_habits": functools.partial(get_user_habits, token),
        "get_habit_detail": functools.partial(get_habit_detail, token),
        "search_habit_research": search_habit_research,
    }


async def _execute_tool_call(block, tool_functions: dict) -> dict:
    try:
        fn = tool_functions[block.name]
        result = await fn(**block.input)
        content = json.dumps(result)
        is_error = False
    except httpx.HTTPStatusError as exc:
        content = f"API returned {exc.response.status_code}: {exc.response.text}"
        is_error = True
    except Exception as exc:
        content = f"Tool call failed: {exc}"
        is_error = True

    return {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": content,
        "is_error": is_error,
    }


def _final_text(response) -> str:
    return "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


async def stream_agent(question: str, token: str, tool_functions: dict | None = None):
    if tool_functions is None:
        tool_functions = build_tool_functions(token)
    messages = [{"role": "user", "content": question}]

    for turn in range(MAX_TURNS):
        async with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    yield {"type": "text", "delta": event.text}
            response = await stream.get_final_message()

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "explanation", None)
            yield {
                "type": "error",
                "code": "refused",
                "detail": detail or "The model declined to answer this request.",
            }
            return

        if response.stop_reason != "tool_use":
            yield {"type": "done", "answer": _final_text(response)}
            return

        calls = [block for block in response.content if block.type == "tool_use"]

        for block in calls:
            yield {"type": "tool_start", "name": block.name, "input": block.input}

        results = await asyncio.gather(
            *(_execute_tool_call(block, tool_functions) for block in calls)
        )

        for block, result in zip(calls, results):
            yield {
                "type": "tool_end",
                "name": block.name,
                "ok": not result["is_error"],
                "detail": result["content"] if result["is_error"] else None,
            }

        messages.append({"role": "user", "content": results})

    yield {
        "type": "error",
        "code": "turn_limit",
        "detail": f"The agent did not reach a final answer within {MAX_TURNS} turns.",
    }


async def run_agent(question: str, token: str, tool_functions: dict | None = None) -> str:
    async for event in stream_agent(question, token, tool_functions=tool_functions):
        if event["type"] == "done":
            return event["answer"]
        if event["type"] == "error":
            if event["code"] == "refused":
                raise AgentRefused(event["detail"])
            raise TurnLimitExceeded(event["detail"])

    raise TurnLimitExceeded("The agent produced no answer.")


async def _cli(question: str, token: str) -> int:
    async for event in stream_agent(question, token):
        if event["type"] == "text":
            print(event["delta"], end="", flush=True)
        elif event["type"] == "tool_start":
            print(f"\n[calling {event['name']} {event['input']}]", flush=True)
        elif event["type"] == "tool_end" and not event["ok"]:
            print(f"[{event['name']} failed: {event['detail']}]", flush=True)
        elif event["type"] == "done":
            print()
            return 0
        elif event["type"] == "error":
            print(f"\n[error] {event['detail']}", file=sys.stderr)
            return 1
    return 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: provide a question. Usage: python agent.py 'your question'")
        sys.exit(1)

    sys.exit(asyncio.run(_cli(sys.argv[1], os.environ["HABIT_API_TOKEN"])))
