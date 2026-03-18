"""
Structured logging configuration for the Durian Classification API.

Provides JSON-formatted log output with configurable log levels,
suitable for production log aggregation systems (ELK, CloudWatch, etc.).
"""

import logging
import sys
from datetime import datetime, timezone
from typing import Any

from core.config import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON log formatter for structured logging.

    Outputs each log record as a single-line JSON object containing
    timestamp, level, logger name, message, and optional exception info.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A JSON-formatted string representation of the log record.
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Include exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields if any
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data

        # Manual JSON serialization to avoid import overhead
        parts = []
        for key, value in log_entry.items():
            if isinstance(value, str):
                # Escape special characters in strings
                escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                parts.append(f'"{key}": "{escaped}"')
            elif isinstance(value, (int, float)):
                parts.append(f'"{key}": {value}')
            elif value is None:
                parts.append(f'"{key}": null')
            else:
                escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                parts.append(f'"{key}": "{escaped}"')

        return "{" + ", ".join(parts) + "}"


def setup_logging() -> None:
    """Configure the root logger with structured JSON output.

    Sets the root logger level from settings and attaches a
    JSONFormatter to the stdout stream handler. Suppresses noisy
    third-party loggers (uvicorn.access, etc.) to WARNING level.
    """
    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers to prevent duplicate output
    root_logger.handlers.clear()

    # Create stdout handler with JSON formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Create a named logger instance.

    Args:
        name: The name for the logger, typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    return logging.getLogger(name)


# Run setup on module import
setup_logging()

# Convenience: pre-configured logger for the core package
logger: logging.Logger = get_logger("durian_ml_service")
