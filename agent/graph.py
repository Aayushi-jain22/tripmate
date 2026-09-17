"""
Agent workflow graph — the framework-based replacement for the old
hand-rolled turn loop in agent/orchestrator.py.

Nodes:
  - "agent": calls the LLM (bound to TOOLS) with the running message list,
    prefixed with the system prompt. The LLM itself decides, per query,
    whether to answer directly or request zero, one, or several tool
    calls -- this is the "dynamic tool selection" the task calls for;
    there is no keyword-routing or if/else tool dispatch anywhere in this
    file.
  - "tools": a prebuilt `ToolNode` that executes whichever tool(s) the
    LLM requested (in one batch if it requested several at once, e.g. a
    packing question needing both the destination guide and the weather
    forecast) and appends their results as ToolMessages.

Edges: START -> "agent" -> (tools_condition) -> "tools" or END;
"tools" -> "agent". `tools_condition` is a LangGraph prebuilt: it checks
whether the latest AIMessage carries tool_calls and routes accordingly.
This is the same "reason -> act -> observe -> repeat" loop the original
orchestrator implemented by hand, expressed declaratively as a graph
instead -- which is what makes it easy to reason about, extend (add a
node, add an edge) and swap/upgrade LLM providers without touching the
control flow.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import AgentState
from agent.tool_schemas import SYSTEM_PROMPT
from tools.langchain_tools import TOOLS


def build_graph(llm: BaseChatModel) -> CompiledStateGraph:
    """Compile the TripMate agent workflow for a given chat model.

    `llm` only needs to satisfy the standard LangChain `BaseChatModel`
    interface (`.bind_tools(...)` -> something with `.invoke(messages)`).
    Swapping providers (Anthropic <-> Ollama <-> anything else LangChain
    supports) never touches this function -- see agent/llm_factory.py.
    """
    llm_with_tools = llm.bind_tools(TOOLS)
    system_message = SystemMessage(content=SYSTEM_PROMPT)

    def agent_node(state: AgentState) -> dict:
        messages = [system_message, *state["messages"]]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
