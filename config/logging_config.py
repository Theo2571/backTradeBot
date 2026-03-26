"""
Structured logging configuration.

Sets up JSON-formatted structured logs when running in production,
and a human-readable format for development/debug use.
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional

import structlog

from .settings import LogLevel, get_settings


def configure_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure structlog + stdlib logging for the application.

    Should be called exactly once at application startup (main.py lifespan).

    Args:
        log_level: Override log level (defaults to settings value).
        log_file:  Optional file path to duplicate log output.
    """
    settings = get_settings()
    level_str = log_level or settings.log_level.value
    level = getattr(logging, level_str, logging.INFO)
    file_path = log_file or settings.log_file

    # ------------------------------------------------------------------ #
    # Build handlers
    # ------------------------------------------------------------------ #
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    # ------------------------------------------------------------------ #
    # stdlib root logger
    # ------------------------------------------------------------------ #
    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    # ------------------------------------------------------------------ #
    # structlog processors
    # ------------------------------------------------------------------ #
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if level == logging.DEBUG:
        # Human-readable in development
        renderer = structlog.dev.ConsoleRenderer()
    else:
        # Machine-readable JSON in production
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    for handler in handlers:
        handler.setFormatter(formatter)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for the given module name."""
    return structlog.get_logger(name)
