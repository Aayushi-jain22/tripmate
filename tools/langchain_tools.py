"""
LangChain tool wrappers around the pure tool implementations.

The actual retrieval/lookup logic lives untouched in tools/rag_tool.py
and tools/weather_tool.py -- this module only adapts those pure
functions to the `@tool` interface LangGraph's ToolNode expects (a
callable with a name, a docstring the LLM sees as the tool description,
and typed arguments it turns into a JSON schema automatically). This
keeps the retrieval logic itself framework-agnostic and independently
unit-testable (see tests/test_rag_tool.py, tests/test_weather_tool.py),
while this thin layer is what the *agentic framework* dispatches to.

Each wrapper also emits the same structured tool_call_started /
tool_call_succeeded / tool_call_error log events the original
hand-rolled orchestrator emitted, so the "visible reasoning trace"
requirement is unaffected by the framework swap -- and converts the
expected typed exceptions (RAGToolError / WeatherToolError) into an
"ERROR: ..." string result instead of letting them propagate, so a bad
tool call never crashes the graph -- the LLM just sees the error and
can explain the gap to the user (see agent/orchestrator.py, which turns
that string back into a ToolCallRecord.error for the trace).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool

from agent.logging_setup import log_event
from tools.rag_tool import RAGToolError
from tools.rag_tool import search_destination_guide as _search_destination_guide
from tools.weather_tool import WeatherToolError
from tools.weather_tool import get_weather_forecast as _get_weather_forecast

_logger = logging.getLogger("tripmate")


@tool
def search_destination_guide(query: str, city_filter: Optional[str] = None) -> str:
    """Search TripMate's destination knowledge base for information about
    visa/entry requirements, best time to visit, local customs and
    etiquette, general packing advice, and safety notes for a destination.
    Currently covers: Tokyo, Reykjavik, Bangkok, and Barcelona.
    Use this for any question about visas, customs, general safety, or what
    a destination is generally like -- but NOT for the current/seasonal
    weather forecast itself (use get_weather_forecast for that).

    Args:
        query: The natural-language question or topic to search for, e.g.
            'do I need a visa for Japan' or 'what to pack for Iceland'.
        city_filter: Optional. Restrict the search to a single known city
            (e.g. 'Tokyo') if the user's query names one.
    """
    log_event(
        _logger, "tool_call_started", tool="search_destination_guide",
        arguments={"query": query, "city_filter": city_filter},
    )
    try:
        results = _search_destination_guide(query, city_filter=city_filter)
    except RAGToolError as exc:
        log_event(_logger, "tool_call_error", level="WARNING", tool="search_destination_guide", error=str(exc))
        return f"ERROR: {exc}"

    if not results:
        log_event(_logger, "tool_call_succeeded", tool="search_destination_guide", result=[])
        return "No relevant information was found for this query."

    log_event(_logger, "tool_call_succeeded", tool="search_destination_guide", result=results)
    return "\n".join(results)


@tool
def get_weather_forecast(city: str, date_or_month: str) -> str:
    """Get the expected weather conditions and temperature range for a city
    during a given month or date. Use this whenever the user asks about
    weather, temperature, or season, or when packing advice needs to
    reflect actual seasonal conditions rather than general tips.

    Args:
        city: City name, e.g. 'Tokyo'.
        date_or_month: A month name, month number, or date, e.g.
            'December', '12', or '2026-12-05'.
    """
    log_event(
        _logger, "tool_call_started", tool="get_weather_forecast",
        arguments={"city": city, "date_or_month": date_or_month},
    )
    try:
        result = _get_weather_forecast(city, date_or_month)
    except WeatherToolError as exc:
        log_event(_logger, "tool_call_error", level="WARNING", tool="get_weather_forecast", error=str(exc))
        return f"ERROR: {exc}"

    log_event(_logger, "tool_call_succeeded", tool="get_weather_forecast", result=result)
    return json.dumps(result)


# The full toolset the agent graph binds to the LLM. Adding a new skill is
# just: write its pure implementation in tools/, add a thin @tool wrapper
# here, and append it to this list -- agent/graph.py and main.py need no
# changes.
TOOLS = [search_destination_guide, get_weather_forecast]
