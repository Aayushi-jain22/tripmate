"""
Provider selection.

This is the *only* place that needs to know which concrete LangChain chat
model backs the agent. agent/graph.py and agent/orchestrator.py depend
only on the standard `BaseChatModel` interface (`.bind_tools()` /
`.invoke()`), so switching TRIPMATE_PROVIDER between "ollama" and
"anthropic" -- or adding a third provider LangChain supports (OpenAI,
Gemini, ...) later -- never touches the graph, the orchestrator, or the
tools.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from agent.config import Settings


class LLMConfigError(Exception):
    """Raised when the configured provider is missing required settings."""


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.ollama_model, base_url=settings.ollama_host, temperature=0)

    if settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise LLMConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your "
                "key, or set TRIPMATE_PROVIDER=ollama to use a local model instead."
            )
        return ChatAnthropic(model=settings.model, api_key=settings.anthropic_api_key, temperature=0)

    raise LLMConfigError(
        f"Unknown TRIPMATE_PROVIDER '{settings.provider}'. Supported values: 'ollama', 'anthropic'."
    )
