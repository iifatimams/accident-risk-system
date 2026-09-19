"""Compact JSON logs; per-frame messages use DEBUG, not INFO."""

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    """Format event context without logging raw sensor payloads."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize structured context."""
        return json.dumps(
            {
                "time": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                **getattr(record, "context", {}),
            }
        )


def configure_logging(level: str = "INFO") -> None:
    """Configure only the package logger."""
    logger = logging.getLogger("accident_risk")
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
