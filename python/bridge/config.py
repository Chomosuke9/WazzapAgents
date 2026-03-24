"""Shared environment-parsing utilities and bridge configuration constants."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Generic env-parsing helpers (used by main, llm1, llm2, log, media)
# ---------------------------------------------------------------------------

def _parse_positive_int(raw: str | None, default: int) -> int:
  if raw is None:
    return default
  try:
    parsed = int(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed > 0 else default


def _parse_positive_float(raw: str | None, default: float) -> float:
  if raw is None:
    return default
  try:
    parsed = float(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed > 0 else default


def _parse_non_negative_int(raw: str | None, default: int) -> int:
  if raw is None:
    return default
  try:
    parsed = int(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed >= 0 else default


def _parse_non_negative_float(raw: str | None, default: float) -> float:
  if raw is None:
    return default
  try:
    parsed = float(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed >= 0 else default


# ---------------------------------------------------------------------------
# Bridge-level configuration constants (previously in main.py)
# ---------------------------------------------------------------------------

HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))
INCOMING_DEBOUNCE_SECONDS = _parse_positive_float(
  os.getenv("INCOMING_DEBOUNCE_SECONDS"), 5.0
)
INCOMING_BURST_MAX_SECONDS = _parse_positive_float(
  os.getenv("INCOMING_BURST_MAX_SECONDS"), 20.0
)
SLOW_BATCH_LOG_MS = _parse_non_negative_int(os.getenv("BRIDGE_SLOW_BATCH_LOG_MS"), 2000)
MAX_TRIGGER_BATCH_AGE_MS = _parse_non_negative_int(
  os.getenv("BRIDGE_MAX_TRIGGER_BATCH_AGE_MS"), 45000
)
REPLY_DEDUP_WINDOW_MS = _parse_non_negative_int(
  os.getenv("BRIDGE_REPLY_DEDUP_WINDOW_MS"), 120000
)
REPLY_DEDUP_MIN_CHARS = _parse_non_negative_int(
  os.getenv("BRIDGE_REPLY_DEDUP_MIN_CHARS"), 24
)
ASSISTANT_ECHO_MERGE_WINDOW_MS = _parse_non_negative_int(
  os.getenv("BRIDGE_ASSISTANT_ECHO_MERGE_WINDOW_MS"), 180000
)
