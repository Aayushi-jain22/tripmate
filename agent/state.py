"""
Graph state schema.

LangGraph passes this typed dict between nodes. We use the standard
"chat agent" shape -- a single running message list -- since that's all
this agent needs: no cross-turn memory beyond the current run, no extra
scratch fields. `add_messages` is LangGraph's reducer for this field: each
node returns only the *new* messages it produced, and the reducer appends
them to the running list rather than the node having to resend the whole
history every time.
"""
from __future__ import annotations

from typing import Annotated, Sequence

from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
