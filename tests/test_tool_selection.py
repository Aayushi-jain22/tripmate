"""
These tests validate the AGENT GRAPH's orchestration logic (routing
between the "agent" and "tools" nodes, trace building, message
threading) by scripting what the LLM "decides" to do via FakeChatModel
-- they never call a real provider. Whether the *real* LLM makes the
same tool-selection decisions on live traffic is exercised manually via
the example runs in the README / demo video, since that depends on live
model behavior, not code under our control.

Note: because these run through the real compiled graph, any tool the
LLM is scripted to call still executes for real against
tools/rag_tool.py / tools/weather_tool.py (see test_integration.py for a
test that additionally asserts on the *content* of those real results).
"""
from __future__ import annotations

import logging

from agent.orchestrator import TripMateAgent
from tests.fakes import FakeChatModel, ai_text, ai_tool_call, ai_tool_calls


def make_agent(scripted_responses):
    llm = FakeChatModel(scripted_responses)
    logger = logging.getLogger("test")
    return TripMateAgent(llm=llm, logger=logger, max_turns=6)


def test_no_tool_needed_for_small_talk():
    # LLM answers directly, no tool_calls at all.
    agent = make_agent([ai_text("I'm doing great, thanks for asking!")])

    response = agent.run("Hey, how are you?")

    assert response.trace == []
    assert "great" in response.answer.lower()


def test_single_tool_selected_for_visa_question():
    agent = make_agent([
        ai_tool_call("search_destination_guide", {"query": "visa for Japan", "city_filter": "Tokyo"}, "t1"),
        ai_text("You'll need a visa to enter Japan as an Indian passport holder."),
    ])

    response = agent.run("Do I need a visa to visit Tokyo?")

    assert len(response.trace) == 1
    assert response.trace[0].tool_name == "search_destination_guide"
    assert response.trace[0].error is None
    assert "visa" in response.answer.lower()


def test_multi_tool_selected_for_packing_question():
    # Both tools requested in a single AIMessage -- the LLM decided it
    # needs the destination guide AND the weather forecast together.
    agent = make_agent([
        ai_tool_calls([
            ("search_destination_guide", {"query": "packing tips", "city_filter": "Tokyo"}, "t1"),
            ("get_weather_forecast", {"city": "Tokyo", "date_or_month": "December"}, "t2"),
        ]),
        ai_text("Pack warm layers -- Tokyo in December is cold with light snow."),
    ])

    response = agent.run("What should I pack for Tokyo in December?")

    assert len(response.trace) == 2
    tool_names = {r.tool_name for r in response.trace}
    assert tool_names == {"search_destination_guide", "get_weather_forecast"}
    assert all(r.error is None for r in response.trace)


def test_tool_error_is_surfaced_in_trace_not_raised():
    agent = make_agent([
        ai_tool_call("get_weather_forecast", {"city": "Atlantis", "date_or_month": "June"}, "t1"),
        ai_text("I don't have weather data for Atlantis, sorry!"),
    ])

    response = agent.run("What's the weather in Atlantis in June?")

    assert len(response.trace) == 1
    assert response.trace[0].error is not None
    assert "atlantis" in response.answer.lower()


def test_out_of_scope_request_produces_no_tool_calls():
    agent = make_agent([
        ai_text("I can't book flights for you -- I can only help with trip planning info."),
    ])

    response = agent.run("Can you book my flight to Paris?")

    assert response.trace == []
    assert "can't" in response.answer.lower() or "cannot" in response.answer.lower()
