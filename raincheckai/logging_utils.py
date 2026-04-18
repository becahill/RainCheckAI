"""Structured logging helpers for RainCheckAI."""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "raincheck_request_id",
    default=None,
)

_RESERVED_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def _json_default(value: Any) -> Any:
    """Convert common Python objects to JSON-serializable values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


class JsonFormatter(logging.Formatter):
    """Render Python log records as single-line JSON payloads."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = _REQUEST_ID.get()
        if request_id is not None:
            payload["request_id"] = request_id

        context = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_KEYS and not key.startswith("_")
        }
        if context:
            payload["context"] = context

        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=_json_default, sort_keys=True)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure global structured logging for the application."""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)


def set_request_id(request_id: str) -> contextvars.Token[str | None]:
    """Bind a request identifier to the current execution context."""
    return _REQUEST_ID.set(request_id)


def reset_request_id(token: contextvars.Token[str | None]) -> None:
    """Restore the previous request identifier binding."""
    _REQUEST_ID.reset(token)
