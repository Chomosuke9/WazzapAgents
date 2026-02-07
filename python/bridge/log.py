from __future__ import annotations

import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv


LOG_RECORD_BUILTINS = {
  "name",
  "msg",
  "args",
  "levelname",
  "levelno",
  "pathname",
  "filename",
  "module",
  "exc_info",
  "exc_text",
  "stack_info",
  "lineno",
  "funcName",
  "created",
  "msecs",
  "relativeCreated",
  "thread",
  "threadName",
  "processName",
  "process",
  "message",
}


def _level_from_env() -> int:
  load_dotenv()
  level = os.getenv("BRIDGE_LOG_LEVEL", "INFO").upper()
  return getattr(logging, level, logging.INFO)


class ExtraFormatter(logging.Formatter):
  """Formatter that appends non-standard record attributes as JSON."""

  def format(self, record: logging.LogRecord) -> str:
    base = super().format(record)
    # Append extras for INFO and below so structured context shows up in normal logs
    if record.levelno <= logging.INFO:
      extras = {k: v for k, v in record.__dict__.items() if k not in LOG_RECORD_BUILTINS}
      if extras:
        base = f"{base} | extras={dump_json(extras, limit=4000)}"
    return base


def setup_logging() -> logging.Logger:
  logger = logging.getLogger("bridge")
  if logger.handlers:
    return logger
  logger.setLevel(_level_from_env())
  handler = logging.StreamHandler(sys.stdout)
  formatter = ExtraFormatter(
    fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
  )
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  logger.propagate = False
  return logger


def trunc(value: Any, limit: int = 400) -> str:
  """Stringify and truncate long values for debug logging."""
  text = str(value)
  if len(text) > limit:
    return text[:limit] + f"...[{len(text) - limit} more]"
  return text


def dump_json(obj: Any, limit: int = 4000) -> str:
  """Safe JSON dump with truncation for large payloads."""
  import json  # local import to keep module light

  try:
    text = json.dumps(obj, ensure_ascii=False, default=str)
  except Exception:
    text = str(obj)
  return trunc(text, limit)
