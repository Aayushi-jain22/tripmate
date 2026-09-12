"""
Lightweight stand-ins for the Anthropic SDK's response objects, so tests
can drive TripMateAgent through specific tool-selection scenarios
without making real API calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class FakeMessage:
    content: list
    stop_reason: str


class FakeMessagesEndpoint:
    """Replaces client.messages -- .create() pops the next canned response."""

    def __init__(self, scripted_responses: list[FakeMessage]):
        self._responses = list(scripted_responses)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> FakeMessage:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeMessagesEndpoint ran out of scripted responses")
        return self._responses.pop(0)


class FakeAnthropicClient:
    def __init__(self, scripted_responses: list[FakeMessage]):
        self.messages = FakeMessagesEndpoint(scripted_responses)
