"""SubAgent environment configuration."""
from __future__ import annotations

import os

try:
  from ..config import _parse_positive_float, _parse_non_negative_int
except ImportError:
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
  from bridge.config import _parse_positive_float, _parse_non_negative_int  # type: ignore

SUBAGENT_URL = os.getenv("SUBAGENT_URL", "http://localhost:5000")
SUBAGENT_WEBHOOK_PORT = _parse_non_negative_int(os.getenv("SUBAGENT_WEBHOOK_PORT"), 8081)
SUBAGENT_WEBHOOK_URL = os.getenv(
  "SUBAGENT_WEBHOOK_URL",
  f"http://localhost:{SUBAGENT_WEBHOOK_PORT}/subagent/callback",
)
SUBAGENT_POLL_INTERVAL = _parse_positive_float(os.getenv("SUBAGENT_POLL_INTERVAL"), 5.0)
SUBAGENT_MAX_POLL_ATTEMPTS = _parse_non_negative_int(os.getenv("SUBAGENT_MAX_POLL_ATTEMPTS"), 120)
SUBAGENT_ENABLED_DEFAULT = os.getenv("SUBAGENT_ENABLED_DEFAULT", "false").lower() == "true"

# Submit retry tunables — used by SubAgentClient.submit() so transient
# rate-limits / 5xx / network blips don't immediately fail the whole task.
SUBAGENT_SUBMIT_RETRY_MAX = _parse_non_negative_int(os.getenv("SUBAGENT_SUBMIT_RETRY_MAX"), 3)
SUBAGENT_SUBMIT_RETRY_BASE_BACKOFF = _parse_positive_float(
  os.getenv("SUBAGENT_SUBMIT_RETRY_BASE_BACKOFF"), 1.0
)
SUBAGENT_SUBMIT_RETRY_MAX_BACKOFF = _parse_positive_float(
  os.getenv("SUBAGENT_SUBMIT_RETRY_MAX_BACKOFF"), 30.0
)
SUBAGENT_HTTP_TIMEOUT = _parse_positive_float(os.getenv("SUBAGENT_HTTP_TIMEOUT"), 30.0)

# How long to wait on the completion webhook before falling back to polling.
# Lowered from 600s to 120s by default — at 600s the per-chat lock was held
# for the full window even if SubAgent silently failed, blocking the chat.
SUBAGENT_WAIT_TIMEOUT_S = _parse_positive_float(os.getenv("SUBAGENT_WAIT_TIMEOUT_S"), 120.0)

# Bounds for context that gets fed back to LLM2 so a noisy sub-agent
# cannot blow up the context window of subsequent turns.
SUBAGENT_REPORT_MAX_CHARS = _parse_non_negative_int(os.getenv("SUBAGENT_REPORT_MAX_CHARS"), 4096)
SUBAGENT_PROGRESS_DETAIL_MAX_CHARS = _parse_non_negative_int(
  os.getenv("SUBAGENT_PROGRESS_DETAIL_MAX_CHARS"), 500
)
