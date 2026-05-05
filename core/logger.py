# core/logger.py

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from core.config import settings


class JSONFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "module":    record.module,
            "function":  record.funcName,
            "line":      record.lineno,
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data

        # json.dumps handles all escaping — unicode, \n, \t, nested dicts, etc.
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging() -> None:
    log_level = getattr(logging, settings.LOG_LEVEL, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


setup_logging()

logger: logging.Logger = get_logger("durian_ml_service")