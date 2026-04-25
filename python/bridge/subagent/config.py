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
