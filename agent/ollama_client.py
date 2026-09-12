"""
Ollama adapter.

Anthropic's SDK exposes `client.messages.create(model=, max_tokens=,
system=, tools=, messages=)` and returns an object with `.content`
(list of blocks with `.type`, and either `.text` or `.name`/`.input`/`.id`)
and `.stop_reason`.

This class exposes the exact same surface but talks to a local Ollama
server instead, so agent/orchestrator.py works identically regardless
of which provider is configured -- only main.py decides which client
to build.
"""
from __future__ import annotations

import uuid
from typing import Any

import ollama

from agent.llm_types import NormalizedMessage, TextBlock, ToolUseBlock


class OllamaConnectionError(Exception):
    """Raised when the local Ollama server can't be reached or errors out."""


def _to_ollama_tools(anthropic_style_tools: list[dict]) -> list[dict]:
    """Convert our Anthropic-shaped tool schemas (agent/tool_schemas.py)
    into the OpenAI-style function-calling format Ollama expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in anthropic_style_tools
    ]


def _to_ollama_messages(system: str, messages: list[dict]) -> list[dict]:
    """Convert the Anthropic-shaped conversation history that
    orchestrator.py builds up into Ollama's chat message format."""
    out: list[dict] = [{"role": "system", "content": system}]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "assistant":
            text_parts = [b.text for b in content if getattr(b, "type", None) == "text"]
            tool_calls = [
                {"function": {"name": b.name, "arguments": b.input}}
                for b in content if getattr(b, "type", None) == "tool_use"
            ]
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
            continue

        for block in content:
            out.append({"role": "tool", "content": block["content"]})

    return out


def _to_normalized(ollama_response: dict) -> NormalizedMessage:
    message = ollama_response.get("message", {})
    tool_calls = message.get("tool_calls") or []

    if tool_calls:
        blocks = [
            ToolUseBlock(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=call["function"]["name"],
                input=call["function"].get("arguments", {}) or {},
            )
            for call in tool_calls
        ]
        return NormalizedMessage(content=blocks, stop_reason="tool_use")

    text = message.get("content", "") or "(no response generated)"
    return NormalizedMessage(content=[TextBlock(text=text)], stop_reason="end_turn")


class _OllamaMessagesEndpoint:
    def __init__(self, model: str, host: str):
        self._model = model
        self._client = ollama.Client(host=host)

    def create(self, *, model: str, max_tokens: int, system: str, tools: list[dict], messages: list[dict]) -> NormalizedMessage:
        ollama_messages = _to_ollama_messages(system, messages)
        ollama_tools = _to_ollama_tools(tools)
        try:
            response = self._client.chat(
                model=self._model,
                messages=ollama_messages,
                tools=ollama_tools,
                options={"num_predict": max_tokens},
            )
        except Exception as exc:
            raise OllamaConnectionError(
                f"Could not reach Ollama at the configured host, or the model "
                f"'{self._model}' isn't pulled yet. Run `ollama serve` and "
                f"`ollama pull {self._model}` first. Original error: {exc}"
            ) from exc
        return _to_normalized(response)


class OllamaClient:
    """Drop-in stand-in for `anthropic.Anthropic()` backed by a local Ollama server."""

    def __init__(self, model: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        self.messages = _OllamaMessagesEndpoint(model=model, host=host)