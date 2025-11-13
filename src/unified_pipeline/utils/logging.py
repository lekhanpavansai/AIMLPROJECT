from __future__ import annotations

import logging
from typing import Optional

import structlog


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure stdlib logging and structlog."""

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[logging.StreamHandler()],
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger with optional name binding."""

    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger=name)
    return logger


