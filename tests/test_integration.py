"""
Integration test: exercises the full multi-tool flow end-to-end.

The LLM's *decisions* are scripted (see test_tool_selection.py for why),
but this test lets the REAL rag_tool.py and weather_tool.py execute --
so it verifies the whole chain: agent receives tool_use request -> agent
calls the real tool -> real tool hits the real data -> result is fed
back to the LLM -> final answer is returned with a correct trace.
"""
import logging

from agent.orchestrator import TripMateAgent
from tests.fakes import FakeAnthropicClient, FakeMessage, FakeTextBlock, FakeToolUseBlock


def test_end_to_end_packing_query_multi_tool_flow():
    scripted = [
        FakeMessage(
            content=[
                FakeToolUseBlock(
                    id="t1", name="search_destination_guide",
                    input={"query": "packing tips for Iceland", "city_filter": "Reykjavik"},
                ),
                FakeToolUseBlock(
                    id="t2", name="get_weather_forecast",
                    input={"city": "Reykjavik", "date_or_month": "January"},
                ),
            ],
            stop_reason="tool_use",
        ),
        FakeMessage(
            content=[FakeTextBlock(
                text="Pack heavy thermal layers and waterproof boots -- Reykjavik in January "
                     "is around -3 to 3C with wind, and the guide recommends windproof outer layers."
            )],
            stop_reason="end_turn",
        ),
    ]

    client = FakeAnthropicClient(scripted)
    logger = logging.getLogger("test")
    agent = TripMateAgent(api_key="unused", model="test-model", logger=logger, client=client)

    response = agent.run("What should I pack for Reykjavik in January?")

    # Both tools ran for real and returned real data.
    assert len(response.trace) == 2

    rag_record = next(r for r in response.trace if r.tool_name == "search_destination_guide")
    weather_record = next(r for r in response.trace if r.tool_name == "get_weather_forecast")

    assert rag_record.error is None
    assert any("pack" in chunk.lower() or "layer" in chunk.lower() for chunk in rag_record.result)

    assert weather_record.error is None
    assert weather_record.result["city"] == "Reykjavik"
    assert weather_record.result["month"] == "January"
    assert weather_record.result["temp_range_c"] == [-3, 3]

    # Final synthesized answer references both sources' substance.
    assert "thermal" in response.answer.lower() or "waterproof" in response.answer.lower()
    assert "-3" in response.answer or "3c" in response.answer.lower() or "3°c" in response.answer.lower()

    # The second call to the LLM included the real tool results in its messages.
    second_call_messages = client.messages.calls[1]["messages"]
    tool_result_message = second_call_messages[-1]
    assert tool_result_message["role"] == "user"
    contents = tool_result_message["content"]
    assert len(contents) == 2
    assert all(block["type"] == "tool_result" for block in contents)
