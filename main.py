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
from agent.logging_setup import configure_logging
from agent.orchestrator import TripMateAgent, AgentError


def build_agent(settings, logger):
    """Construct the right LLM client based on TRIPMATE_PROVIDER and hand
    it to TripMateAgent. This is the only place that needs to know which
    provider is active -- the orchestrator itself is provider-agnostic."""
    if settings.provider == "ollama":
        from agent.ollama_client import OllamaClient

        client = OllamaClient(model=settings.ollama_model, host=settings.ollama_host)
        return TripMateAgent(
            api_key="unused-for-ollama",
            model=settings.ollama_model,
            logger=logger,
            client=client,
        )

    return TripMateAgent(
        api_key=settings.anthropic_api_key,
        model=settings.model,
        logger=logger,
    )


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