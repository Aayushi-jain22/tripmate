"""
Central configuration for TripMate.

All configuration is pulled from environment variables (loaded from a
.env file if present) so that no secrets or environment-specific values
are hardcoded anywhere in the codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load a .env file if one exists in the current working directory.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    provider: str  # "anthropic" or "ollama"
    anthropic_api_key: str | None
    model: str
    ollama_model: str
    ollama_host: str
    log_level: str
    max_agent_turns: int = 6  # safety cap on the tool-calling loop


def get_settings() -> Settings:
    return Settings(
        provider=os.getenv("TRIPMATE_PROVIDER", "anthropic").strip().lower(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        model=os.getenv("TRIPMATE_MODEL", "claude-3-5-sonnet-20241022"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        log_level=os.getenv("TRIPMATE_LOG_LEVEL", "INFO"),
    )