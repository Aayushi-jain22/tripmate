"""
Integration test: exercises the full multi-tool flow end-to-end through
the compiled LangGraph.

The LLM's *decisions* are scripted (see test_tool_selection.py for why),
but this test lets the REAL rag_tool.py and weather_tool.py execute via
LangGraph's ToolNode -- so it verifies the whole chain: agent node
requests two tool calls in one turn -> ToolNode runs both real tools
against real data -> both results are fed back to the agent node as
ToolMessages -> the (fake, scripted) LLM synthesizes a final answer ->
TripMateAgent reconstructs a trace with the real tool output attached.
"""
from __future__ import annotations

import json
import logging

from agent.orchestrator import TripMateAgent
from tests.fakes import FakeChatModel, ai_text, ai_tool_calls


def test_end_to_end_packing_query_multi_tool_flow():
    llm = FakeChatModel([
        ai_tool_calls([
            ("search_destination_guide", {"query": "packing tips for Iceland", "city_filter": "Reykjavik"}, "t1"),
            ("get_weather_forecast", {"city": "Reykjavik", "date_or_month": "January"}, "t2"),
        ]),
        ai_text(
            "Pack heavy thermal layers and waterproof boots -- Reykjavik in January "
            "is around -3 to 3C with wind, and the guide recommends windproof outer layers."
        ),
    ])
    logger = logging.getLogger("test")
    agent = TripMateAgent(llm=llm, logger=logger, max_turns=6)

    response = agent.run("What should I pack for Reykjavik in January?")

    # Both tools ran for real and returned real data.
    assert len(response.trace) == 2

    rag_record = next(r for r in response.trace if r.tool_name == "search_destination_guide")
    weather_record = next(r for r in response.trace if r.tool_name == "get_weather_forecast")

    assert rag_record.error is None
    assert "pack" in rag_record.result.lower() or "layer" in rag_record.result.lower()

    assert weather_record.error is None
    weather_data = json.loads(weather_record.result)
    assert weather_data["city"] == "Reykjavik"
    assert weather_data["month"] == "January"
    assert weather_data["temp_range_c"] == [-3, 3]

    # Final synthesized answer references both sources' substance.
    assert "thermal" in response.answer.lower() or "waterproof" in response.answer.lower()
    assert "-3" in response.answer or "3c" in response.answer.lower() or "3°c" in response.answer.lower()

    # The agent node's second call to the LLM included both real tool
    # results as ToolMessages.
    second_call_messages = llm.invocations[1]
    tool_messages = [m for m in second_call_messages if getattr(m, "type", None) == "tool"]
    assert len(tool_messages) == 2
