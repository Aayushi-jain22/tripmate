"""
Module 1 — Agent Core / Orchestration.

TripMateAgent wraps a raw Anthropic tool-use loop:
  1. Send the user query + tool schemas to the LLM.
  2. If the LLM requests tool call(s), execute them against the real
     tool functions (tools/rag_tool.py, tools/weather_tool.py).
  3. Feed the tool result(s) back to the LLM.
  4. Repeat until the LLM returns a final text-only response (or a
     turn cap is hit, as a safety valve against loops).

This is intentionally framework-free (no LangChain/LangGraph) so the
control flow is fully visible and testable -- see README "Technical
decisions" for the rationale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from agent.tool_schemas import SYSTEM_PROMPT, TOOL_SCHEMAS
from agent.logging_setup import log_event
from tools.rag_tool import search_destination_guide, RAGToolError
from tools.weather_tool import get_weather_forecast, WeatherToolError


@dataclass
class ToolCallRecord:
    """One entry in the visible reasoning trace."""
    tool_name: str
    arguments: dict
    result: Any = None
    error: str | None = None


@dataclass
class AgentResponse:
    answer: str
    trace: list[ToolCallRecord] = field(default_factory=list)
    raw_turns: int = 0


class AgentError(Exception):
    """Raised for agent-level failures (e.g. missing API key, malformed input)."""


# Maps tool name -> (callable, list of expected arg names) so we can
# validate + dispatch generically.
_TOOL_DISPATCH: dict[str, Callable[..., Any]] = {
    "search_destination_guide": lambda args: search_destination_guide(
        query=args.get("query", ""), city_filter=args.get("city_filter")
    ),
    "get_weather_forecast": lambda args: get_weather_forecast(
        city=args.get("city", ""), date_or_month=args.get("date_or_month", "")
    ),
}

_TOOL_EXPECTED_ERRORS = (RAGToolError, WeatherToolError)


class TripMateAgent:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        logger: logging.Logger,
        max_turns: int = 6,
        client: Any = None,
    ):
        """
        `client` is exposed purely for testing (dependency injection of a
        fake Anthropic client so tests don't hit the real API). Production
        callers should omit it and let the agent build a real client from
        `api_key`.
        """
        if client is None:
            if not api_key:
                raise AgentError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self._model = model
        self._logger = logger
        self._max_turns = max_turns

    def run(self, user_query: str) -> AgentResponse:
        if not user_query or not user_query.strip():
            raise AgentError("Query must be a non-empty string.")

        log_event(self._logger, "user_query_received", query=user_query)

        messages: list[dict] = [{"role": "user", "content": user_query}]
        trace: list[ToolCallRecord] = []

        for turn in range(1, self._max_turns + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                )
            except anthropic.APIError as exc:
                log_event(self._logger, "llm_call_failed", level="ERROR", error=str(exc), turn=turn)
                raise AgentError(f"LLM call failed: {exc}") from exc

            except Exception as exc:  # provider-agnostic fallback (e.g. Ollama connection errors)
                log_event(self._logger, "llm_call_failed", level="ERROR", error=str(exc), turn=turn)
                raise AgentError(f"LLM call failed: {exc}") from exc

            log_event(
                self._logger,
                "llm_turn_complete",
                turn=turn,
                stop_reason=response.stop_reason,
            )

            if response.stop_reason != "tool_use":
                final_text = self._extract_text(response)
                log_event(self._logger, "agent_final_response", answer=final_text, turns=turn)
                return AgentResponse(answer=final_text, trace=trace, raw_turns=turn)

            # Model wants to call one or more tools this turn.
            messages.append({"role": "assistant", "content": response.content})
            tool_result_blocks = []

            for block in response.content:
                if block.type != "tool_use":
                    continue
                record = self._execute_tool(block.name, block.input)
                trace.append(record)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": self._format_tool_output(record),
                        "is_error": record.error is not None,
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})

        log_event(self._logger, "agent_turn_cap_reached", level="WARNING", turns=self._max_turns)
        return AgentResponse(
            answer=(
                "I wasn't able to settle on a final answer within my reasoning budget. "
                "Could you rephrase or narrow your question?"
            ),
            trace=trace,
            raw_turns=self._max_turns,
        )

    def _execute_tool(self, tool_name: str, arguments: dict) -> ToolCallRecord:
        log_event(self._logger, "tool_call_started", tool=tool_name, arguments=arguments)

        dispatch = _TOOL_DISPATCH.get(tool_name)
        if dispatch is None:
            error_msg = f"Unknown tool requested by model: '{tool_name}'"
            log_event(self._logger, "tool_call_error", level="ERROR", tool=tool_name, error=error_msg)
            return ToolCallRecord(tool_name=tool_name, arguments=arguments, error=error_msg)

        try:
            result = dispatch(arguments)
        except _TOOL_EXPECTED_ERRORS as exc:
            log_event(self._logger, "tool_call_error", level="WARNING", tool=tool_name, error=str(exc))
            return ToolCallRecord(tool_name=tool_name, arguments=arguments, error=str(exc))
        except Exception as exc:  # unexpected/defensive: never let a tool crash the agent
            log_event(
                self._logger, "tool_call_unexpected_error", level="ERROR",
                tool=tool_name, error=str(exc), error_type=type(exc).__name__,
            )
            return ToolCallRecord(
                tool_name=tool_name, arguments=arguments,
                error=f"Unexpected internal error in tool '{tool_name}': {exc}",
            )

        log_event(self._logger, "tool_call_succeeded", tool=tool_name, result=result)
        return ToolCallRecord(tool_name=tool_name, arguments=arguments, result=result)

    @staticmethod
    def _format_tool_output(record: ToolCallRecord) -> str:
        if record.error is not None:
            return f"ERROR: {record.error}"
        if record.result in (None, [], {}):
            return "No relevant information was found for this query."
        return str(record.result)

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> str:
        parts = [block.text for block in response.content if block.type == "text"]
        return "\n".join(parts).strip() or "(no response generated)"
