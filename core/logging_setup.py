"""Structured JSON logging."""
from __future__ import annotations
import json
import logging
import os
import sys
import time
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any extras attached via logger.info("msg", extra={...})
        for k, v in record.__dict__.items():
            if k in ("args", "asctime", "created", "exc_info", "exc_text",
                     "filename", "funcName", "levelname", "levelno", "lineno",
                     "module", "msecs", "message", "msg", "name", "pathname",
                     "process", "processName", "relativeCreated", "stack_info",
                     "thread", "threadName", "taskName"):
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = str(v)
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def configure_logging(level: str | None = None) -> None:
    global _configured
    if _configured:
        return
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
