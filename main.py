"""
TripMate CLI entry point.

Usage:
    python main.py --query "What should I pack for Tokyo in December?"
    python main.py                     # interactive REPL mode
    python main.py --query "..." --verbose   # print the full reasoning trace
"""
from __future__ import annotations

import argparse
import json
import sys

from agent.config import get_settings
from agent.llm_factory import LLMConfigError, build_chat_model
from agent.logging_setup import configure_logging
from agent.orchestrator import TripMateAgent, AgentError


def build_agent(settings, logger):
    """Build the right LangChain chat model for TRIPMATE_PROVIDER (see
    agent/llm_factory.py) and hand it to TripMateAgent, which compiles the
    LangGraph workflow around it. This is the only place that needs to
    know which provider is active -- the graph and orchestrator are
    provider-agnostic."""
    try:
        llm = build_chat_model(settings)
    except LLMConfigError as exc:
        raise AgentError(str(exc)) from exc

    return TripMateAgent(llm=llm, logger=logger, max_turns=settings.max_agent_turns)


def print_trace(trace) -> None:
    if not trace:
        print("(no tools were called -- answered directly)")
        return
    for i, record in enumerate(trace, start=1):
        print(f"  {i}. tool={record.tool_name} args={json.dumps(record.arguments)}")
        if record.error:
            print(f"     -> ERROR: {record.error}")
        else:
            print(f"     -> result: {record.result}")


def run_single(agent: TripMateAgent, query: str, verbose: bool) -> None:
    try:
        response = agent.run(query)
    except AgentError as exc:
        print(f"Agent error: {exc}")
        return

    print(f"\nQuery: {query}")
    if verbose:
        print("\nReasoning trace:")
        print_trace(response.trace)
    print(f"\nTripMate: {response.answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="TripMate agentic travel assistant")
    parser.add_argument("--query", "-q", type=str, help="Single query to run non-interactively.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print the full tool-call trace.")
    args = parser.parse_args()

    settings = get_settings()
    logger = configure_logging(settings.log_level)

    try:
        agent = build_agent(settings, logger)
    except AgentError as exc:
        print(f"Startup error: {exc}")
        sys.exit(1)

    if args.query:
        run_single(agent, args.query, args.verbose)
        return

    print("TripMate agent (type 'exit' to quit)")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        run_single(agent, query, args.verbose)


if __name__ == "__main__":
    main()