import httpx
from anthropic.types import Message, RefusalStopDetails, TextBlock, ToolUseBlock, Usage

import agent


def _message(content, stop_reason, stop_details=None):
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=agent.MODEL,
        content=content,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def says(text):
    return _message([TextBlock(type="text", text=text)], "end_turn")


def calls_tools(*calls):
    content = [
        ToolUseBlock(type="tool_use", id=f"toolu_{i}", name=name, input=tool_input)
        for i, (name, tool_input) in enumerate(calls)
    ]
    return _message(content, "tool_use")


def refuses(explanation):
    return _message(
        [],
        "refusal",
        RefusalStopDetails(type="refusal", category="general_harms", explanation=explanation),
    )


class FakeMessages:
    def __init__(self, responses, repeat_last):
        self._queue = list(responses)
        self._repeat_last = repeat_last
        self.calls = []

    def create(self, **kwargs):
        kwargs["messages"] = list(kwargs["messages"])
        self.calls.append(kwargs)
        if not self._queue:
            raise AssertionError("the model was called more times than the script allows")
        if self._repeat_last and len(self._queue) == 1:
            return self._queue[0]
        return self._queue.pop(0)


class FakeAnthropic:
    def __init__(self, *responses, repeat_last=False):
        self.messages = FakeMessages(responses, repeat_last)

    def sent_messages(self, call_index):
        return self.messages.calls[call_index]["messages"]


class FakeHabitAPI:
    def __init__(self, base_url):
        self._base_url = base_url
        self._routes = {}
        self.requests = []

    def route(self, path, *, json=None, status=200):
        self._routes[path] = (status, json)

    def _handle(self, request):
        self.requests.append(request)
        if request.url.path not in self._routes:
            return httpx.Response(404, json={"detail": "Not Found"})
        status, payload = self._routes[request.url.path]
        return httpx.Response(status, json=payload)

    def build_client(self):
        return httpx.Client(
            base_url=self._base_url,
            transport=httpx.MockTransport(self._handle),
        )

    @property
    def paths(self):
        return [request.url.path for request in self.requests]

    @property
    def auth_headers(self):
        return [request.headers.get("authorization") for request in self.requests]
