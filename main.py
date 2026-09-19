import sys
import anthropic
from dotenv import load_dotenv

if len(sys.argv) < 2:
    print("Error: provide a question. Usage: python main.py 'your question'")
    sys.exit(1)

question = sys.argv[1]

load_dotenv()
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-5",
    max_tokens=10000,
    system="You are a concise habit-formation coach. Help the user build sustainable routines with brief, actionable advice",
    messages=[
        {"role": "user", "content": question},
    ],
)

for block in response.content:
    if block.type == "text":
        print(block.text)

print(f"\nStop reason: {response.stop_reason}")

print(f"Input tokens: {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")