"""
Agent Core — thin run-loop wrapper around the compiled LangGraph workflow
(agent/graph.py).

TripMateAgent.run(query):
  1. Seeds graph state with the user's message.
  2. Invokes the compiled graph, which internally loops
     agent -> tools -> agent until the LLM returns a final,
     tool-call-free response (or a turn cap is hit, as a safety valve
     against runaway loops).
  3. Reconstructs the same `AgentResponse` / `ToolCallRecord` shape the
     original hand-rolled orchestrator returned, so main.py's CLI and the
     trace format are unaffected by the framework swap underneath.

All actual routing/dispatch logic now lives in the graph itself
(agent/graph.py) and in LangGraph's prebuilt `ToolNode` -- this class
does not decide which tool to call or when; it only drives the graph and
translates its output back into TripMate's own response/trace types.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.graph import build_graph
from agent.logging_setup import log_event


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
    """Raised for agent-level failures (e.g. missing config, malformed input, LLM call failure)."""


class TripMateAgent:
    def __init__(self, llm: BaseChatModel, logger: logging.Logger, max_turns: int = 6):
        self._graph = build_graph(llm)
        self._logger = logger
        self._max_turns = max_turns

    def run(self, user_query: str) -> AgentResponse:
        if not user_query or not user_query.strip():
            raise AgentError("Query must be a non-empty string.")

        log_event(self._logger, "user_query_received", query=user_query)

        try:
            result = self._graph.invoke(
                {"messages": [HumanMessage(content=user_query)]},
                config={"recursion_limit": self._max_turns * 2 + 2},
            )
        except GraphRecursionError:
            log_event(self._logger, "agent_turn_cap_reached", level="WARNING", turns=self._max_turns)
            return AgentResponse(
                answer=(
                    "I wasn't able to settle on a final answer within my reasoning budget. "
                    "Could you rephrase or narrow your question?"
                ),
                raw_turns=self._max_turns,
            )
        except Exception as exc:  # provider-agnostic: network errors, auth errors, etc.
            log_event(self._logger, "llm_call_failed", level="ERROR", error=str(exc))
            raise AgentError(f"LLM call failed: {exc}") from exc

        messages = result["messages"]
        trace = self._build_trace(messages)
        final_text = self._extract_final_text(messages)
        turns = sum(1 for m in messages if isinstance(m, AIMessage))

        log_event(self._logger, "agent_final_response", answer=final_text, turns=turns)
        return AgentResponse(answer=final_text, trace=trace, raw_turns=turns)

    @staticmethod
    def _build_trace(messages: list) -> list[ToolCallRecord]:
        """Pair each requested tool_call with the ToolMessage the ToolNode
        produced for it (matched by tool_call_id) to rebuild the trace."""
        tool_messages_by_id = {
            m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)
        }

        trace: list[ToolCallRecord] = []
        for msg in messages:
            if not (isinstance(msg, AIMessage) and msg.tool_calls):
                continue
            for call in msg.tool_calls:
                tool_msg = tool_messages_by_id.get(call["id"])
                content = tool_msg.content if tool_msg is not None else None

                error: str | None = None
                result: Any = content
                if isinstance(content, str) and content.startswith("ERROR:"):
                    error = content[len("ERROR:"):].strip()
                    result = None

                trace.append(
                    ToolCallRecord(
                        tool_name=call["name"],
                        arguments=call.get("args", {}),
                        result=result,
                        error=error,
                    )
                )
        return trace

    @staticmethod
    def _extract_final_text(messages: list) -> str:
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                return (msg.content or "").strip() or "(no response generated)"
        return "(no response generated)"
