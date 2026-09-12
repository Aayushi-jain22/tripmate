"""
Structured logging for TripMate.

Every tool call, reasoning step, and error is emitted as a single-line
JSON record so traces can be grepped, piped into a log aggregator, or
pasted straight into the README as example runs.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("tripmate")
    logger.setLevel(level.upper())
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger


def log_event(logger: logging.Logger, event: str, level: str = "INFO", **fields: Any) -> None:
    """Emit a single structured log line, e.g.

    log_event(logger, "tool_call", tool="get_weather_forecast", args={...})
    """
    log_fn = getattr(logger, level.lower(), logger.info)
    log_fn(event, extra={"extra_fields": fields})
