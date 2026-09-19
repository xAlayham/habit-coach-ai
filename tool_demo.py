import sys
import anthropic
from dotenv import load_dotenv

if len(sys.argv) < 2:
    print("Error: provide a question. Usage: python main.py 'your question'")
    sys.exit(1)

question = sys.argv[1]

load_dotenv()
client = anthropic.Anthropic()

get_habit_completions = {
    "name": "get_habit_completions",
    "description": (
        "Does a thing"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "habit_name": {
                "type": "string",
                "description": "Habit name, e.g. 'Walk 10k steps'",
            },
            "days": {
                "type": "integer",
                "description": "How many days in a row has he completed said habit"
            },
        },
        "required": ["habit_name"],
    },
}

response = client.messages.create(
    model="claude-opus-5",
    max_tokens=16000,
    tools=[get_habit_completions],
    messages=[{"role": "user", "content": question}],
)

print(f"Stop reason: {response.stop_reason}")

for block in response.content:
    if block.type == "text":
        print(f"[text] {block.text}")
    elif block.type == "tool_use":
        print(f"[tool_use] id  = {block.id}")
        print(f"[tool_use] name  = {block.name}")
        print(f"[tool_use] input  = {block.input}")