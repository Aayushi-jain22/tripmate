"""
These tests validate the AGENT'S ORCHESTRATION LOGIC (dispatch, trace
building, message threading) by scripting what the LLM "decides" to
do -- they do not call the real Anthropic API. Whether the *real* LLM
makes the same tool-selection decisions is exercised manually via the
example runs in the README / Loom video, since that depends on live
model behavior, not code under our control.
"""
import logging

from agent.orchestrator import TripMateAgent
from tests.fakes import FakeAnthropicClient, FakeMessage, FakeTextBlock, FakeToolUseBlock


def make_agent(scripted_responses):
    client = FakeAnthropicClient(scripted_responses)
    logger = logging.getLogger("test")
    return TripMateAgent(api_key="unused", model="test-model", logger=logger, client=client)


def test_no_tool_needed_for_small_talk():
    # LLM answers directly, no tool_use block at all.
    scripted = [
        FakeMessage(content=[FakeTextBlock(text="I'm doing great, thanks for asking!")], stop_reason="end_turn")
    ]
    agent = make_agent(scripted)

    response = agent.run("Hey, how are you?")

    assert response.trace == []
    assert "great" in response.answer.lower()


def test_single_tool_selected_for_visa_question():
    scripted = [
        FakeMessage(
            content=[FakeToolUseBlock(id="t1", name="search_destination_guide",
                                       input={"query": "visa for Japan", "city_filter": "Tokyo"})],
            stop_reason="tool_use",
        ),
        FakeMessage(
            content=[FakeTextBlock(text="You'll need a visa to enter Japan as an Indian passport holder.")],
            stop_reason="end_turn",
        ),
    ]
    agent = make_agent(scripted)

    response = agent.run("Do I need a visa to visit Tokyo?")

    assert len(response.trace) == 1
    assert response.trace[0].tool_name == "search_destination_guide"
    assert response.trace[0].error is None
    assert "visa" in response.answer.lower()


def test_multi_tool_selected_for_packing_question():
    scripted = [
        FakeMessage(
            content=[
                FakeToolUseBlock(id="t1", name="search_destination_guide",
                                  input={"query": "packing tips", "city_filter": "Tokyo"}),
                FakeToolUseBlock(id="t2", name="get_weather_forecast",
                                  input={"city": "Tokyo", "date_or_month": "December"}),
            ],
            stop_reason="tool_use",
        ),
        FakeMessage(
            content=[FakeTextBlock(text="Pack warm layers -- Tokyo in December is cold with light snow.")],
            stop_reason="end_turn",
        ),
    ]
    agent = make_agent(scripted)

    response = agent.run("What should I pack for Tokyo in December?")

    assert len(response.trace) == 2
    tool_names = {r.tool_name for r in response.trace}
    assert tool_names == {"search_destination_guide", "get_weather_forecast"}
    assert all(r.error is None for r in response.trace)


def test_tool_error_is_surfaced_in_trace_not_raised():
    scripted = [
        FakeMessage(
            content=[FakeToolUseBlock(id="t1", name="get_weather_forecast",
                                       input={"city": "Atlantis", "date_or_month": "June"})],
            stop_reason="tool_use",
        ),
        FakeMessage(
            content=[FakeTextBlock(text="I don't have weather data for Atlantis, sorry!")],
            stop_reason="end_turn",
        ),
    ]
    agent = make_agent(scripted)

    response = agent.run("What's the weather in Atlantis in June?")

    assert len(response.trace) == 1
    assert response.trace[0].error is not None
    assert "atlantis" in response.answer.lower()


def test_out_of_scope_request_produces_no_tool_calls():
    scripted = [
        FakeMessage(
            content=[FakeTextBlock(text="I can't book flights for you -- I can only help with trip planning info.")],
            stop_reason="end_turn",
        )
    ]
    agent = make_agent(scripted)

    response = agent.run("Can you book my flight to Paris?")

    assert response.trace == []
    assert "can't" in response.answer.lower() or "cannot" in response.answer.lower()
