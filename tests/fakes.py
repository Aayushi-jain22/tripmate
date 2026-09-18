"""
Fake chat model for testing the agent graph without a real LLM provider.

`FakeChatModel` implements just enough of LangChain's `BaseChatModel`
surface (`.bind_tools(...)` -> object with `.invoke(messages)`) for
agent/graph.py's agent_node to drive it -- each `.invoke()` call pops the
next scripted `AIMessage` in order. This lets tests script exactly what
the LLM "decides" to do on each turn (answer directly, request one tool,
request several at once) while the REAL tool implementations still run
underneath via LangGraph's `ToolNode` -- so these fakes only replace the
non-deterministic part (the model's judgment call), never the tool logic
itself.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage


class FakeChatModel:
    def __init__(self, scripted_responses: list[AIMessage]):
        self._responses = list(scripted_responses)
        self.invocations: list[list] = []

    def bind_tools(self, tools):
        # Real chat models return a new runnable bound to these tools;
        # the fake has no schema-generation step to do, so it just hands
        # itself back.
        return self

    def invoke(self, messages, **kwargs):
        self.invocations.append(list(messages))
        if not self._responses:
            raise AssertionError("FakeChatModel ran out of scripted responses")
        return self._responses.pop(0)


def ai_text(text: str) -> AIMessage:
    """A scripted final answer -- no tool calls, ends the graph run."""
    return AIMessage(content=text)


def ai_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    """A scripted turn requesting exactly one tool call."""
    return ai_tool_calls([(name, args, call_id)])


def ai_tool_calls(calls: list[tuple[str, dict, str]]) -> AIMessage:
    """A scripted turn requesting one or more tool calls in a single
    batch (e.g. both search_destination_guide and get_weather_forecast
    for a packing question) -- LangGraph's ToolNode executes every
    tool_call on an AIMessage before routing back to the agent node."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
            for name, args, call_id in calls
        ],
    )
