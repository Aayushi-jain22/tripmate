"""
Provider-agnostic response shapes.

Both the real Anthropic SDK response and our OllamaClient adapter (see
ollama_client.py) expose objects duck-typed to this shape, so
agent/orchestrator.py never needs to know which provider is behind
`self._client`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class NormalizedMessage:
    content: list
    stop_reason: str  # "tool_use" or "end_turn"