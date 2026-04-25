from __future__ import annotations

import asyncio
import time
from typing import Optional

try:
  import requests
except ImportError:
  requests = None  # type: ignore

try:
  from .config import (
    SUBAGENT_URL,
    SUBAGENT_WEBHOOK_URL,
    SUBAGENT_POLL_INTERVAL,
    SUBAGENT_MAX_POLL_ATTEMPTS,
  )
except ImportError:
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
  from bridge.subagent.config import (  # type: ignore
    SUBAGENT_URL,
    SUBAGENT_WEBHOOK_URL,
    SUBAGENT_POLL_INTERVAL,
    SUBAGENT_MAX_POLL_ATTEMPTS,
  )


class SubAgentClient:
  def __init__(
    self,
    base_url: str | None = None,
    webhook_url: str | None = None,
  ) -> None:
    self._base_url = (base_url or SUBAGENT_URL).rstrip("/")
    self._webhook_url = webhook_url or SUBAGENT_WEBHOOK_URL

  async def submit(
    self,
    session_id: str,
    instruction: str,
    input_files: list[str],
  ) -> dict:
    """Submit a task to the SubAgent (non-blocking)."""
    loop = asyncio.get_running_loop()
    payload = {
      "session_id": session_id,
      "instruction": instruction,
      "input_files": input_files,
      "callback_url": self._webhook_url,
      "progress_webhook": f"{self._webhook_url}?type=progress",
    }
    return await loop.run_in_executor(
      None,
      lambda: self._post_sync(f"{self._base_url}/execute", payload),
    )

  def _post_sync(self, url: str, payload: dict) -> dict:
    if requests is None:
      raise RuntimeError("requests library is not installed")
    resp = requests.post(url, json=payload, timeout=30)
    # Return response JSON if available; otherwise basic status info
    try:
      body = resp.json()
    except Exception:
      body = {"status_code": resp.status_code, "text": resp.text}
    body["_status_code"] = resp.status_code
    return body

  async def poll_result(self, session_id: str) -> dict | None:
    """Poll SubAgent for result. Returns result dict or None on exhaustion."""
    url = f"{self._base_url}/sessions/{session_id}/result"
    for attempt in range(1, SUBAGENT_MAX_POLL_ATTEMPTS + 1):
      loop = asyncio.get_running_loop()
      resp = await loop.run_in_executor(None, lambda: self._get_sync(url))
      if resp.get("_status_code") == 200:
        data = resp.get("result") or resp
        # If the endpoint returns a completed result, return it
        if isinstance(data, dict) and data.get("success") is not None:
          return data
      await asyncio.sleep(SUBAGENT_POLL_INTERVAL)
    return None

  def _get_sync(self, url: str) -> dict:
    if requests is None:
      raise RuntimeError("requests library is not installed")
    resp = requests.get(url, timeout=30)
    try:
      body = resp.json()
    except Exception:
      body = {"status_code": resp.status_code, "text": resp.text}
    body["_status_code"] = resp.status_code
    return body