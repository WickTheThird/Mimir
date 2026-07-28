"""Structured logging with correlation IDs and secret redaction (ADR 20)."""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import structlog

from mimir.redaction import redact

_correlation_id: ContextVar[str | None] = ContextVar("mimir_correlation_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("mimir_session_id", default=None)
_configured = False


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def get_session_id() -> str | None:
    return _session_id.get()


@contextmanager
def correlation_context(
    correlation_id: str | None = None, session_id: str | None = None
) -> Iterator[str]:
    cid = correlation_id or new_correlation_id()
    token = _correlation_id.set(cid)
    session_token = _session_id.set(session_id) if session_id else None
    try:
        yield cid
    finally:
        _correlation_id.reset(token)
        if session_token is not None:
            _session_id.reset(session_token)


def _add_context(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    cid = _correlation_id.get()
    if cid:
        event_dict.setdefault("correlation_id", cid)
    sid = _session_id.get()
    if sid:
        event_dict.setdefault("session_id", sid)
    return event_dict


def _redact_processor(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            event_dict[key] = redact(value)
    return event_dict


def configure_logging(
    level: str = "INFO",
    json_logs: bool = False,
    log_file: Path | None = None,
    force: bool = False,
) -> None:
    global _configured
    if _configured and not force:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_context,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_processor,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # Uvicorn access logs are noisy for a single-user local service.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str = "mimir") -> Any:
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
