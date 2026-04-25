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
    SUBAGENT_SUBMIT_RETRY_MAX,
    SUBAGENT_SUBMIT_RETRY_BASE_BACKOFF,
    SUBAGENT_SUBMIT_RETRY_MAX_BACKOFF,
    SUBAGENT_HTTP_TIMEOUT,
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
    SUBAGENT_SUBMIT_RETRY_MAX,
    SUBAGENT_SUBMIT_RETRY_BASE_BACKOFF,
    SUBAGENT_SUBMIT_RETRY_MAX_BACKOFF,
    SUBAGENT_HTTP_TIMEOUT,
  )


class SubAgentSubmitError(RuntimeError):
  """Raised when /execute fails after retries.

  Carries ``status_code`` (HTTP status of the last response, if any) and
  ``body`` (raw JSON or text from the SubAgent) so callers can log and
  surface a clean error to the user instead of waiting indefinitely on a
  webhook that will never arrive.
  """

  def __init__(self, message: str, status_code: int | None = None, body: dict | None = None) -> None:
    super().__init__(message)
    self.status_code = status_code
    self.body = body or {}


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
    """Submit a task to the SubAgent (non-blocking).

    Retries transient submit errors (network blip, 429, 5xx) with
    exponential backoff. Raises :class:`SubAgentSubmitError` on permanent
    failure or after all retries are exhausted, so the caller does not
    silently wait for a webhook that will never arrive.
    """
    payload = {
      "session_id": session_id,
      "instruction": instruction,
      "input_files": input_files,
      "callback_url": self._webhook_url,
      "progress_webhook": f"{self._webhook_url}?type=progress",
    }
    url = f"{self._base_url}/execute"
    attempts = max(1, SUBAGENT_SUBMIT_RETRY_MAX + 1)
    last_status: int | None = None
    last_body: dict = {}
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
      loop = asyncio.get_running_loop()
      try:
        body = await loop.run_in_executor(None, lambda: self._post_sync(url, payload))
      except Exception as err:  # network failure, JSON decode error, etc.
        last_err = err
        last_status = None
        last_body = {}
        if attempt >= attempts:
          raise SubAgentSubmitError(
            f"Failed to reach SubAgent at {url} after {attempts} attempts: {err}",
          ) from err
        await asyncio.sleep(_backoff_seconds(attempt))
        continue

      status = body.get("_status_code")
      last_status = status if isinstance(status, int) else None
      last_body = body
      if isinstance(status, int) and 200 <= status < 300:
        return body
      retryable = isinstance(status, int) and (status == 429 or 500 <= status < 600)
      if not retryable or attempt >= attempts:
        raise SubAgentSubmitError(
          f"SubAgent /execute returned status={status}",
          status_code=last_status,
          body=last_body,
        )
      await asyncio.sleep(_backoff_seconds(attempt, body=body))

    # Defensive — loop above should always either return or raise.
    raise SubAgentSubmitError(
      f"SubAgent submit loop exhausted without a result (last_err={last_err})",
      status_code=last_status,
      body=last_body,
    )

  def _post_sync(self, url: str, payload: dict) -> dict:
    if requests is None:
      raise RuntimeError("requests library is not installed")
    resp = requests.post(url, json=payload, timeout=SUBAGENT_HTTP_TIMEOUT)
    # Return response JSON if available; otherwise basic status info
    try:
      body = resp.json()
    except Exception:
      body = {"status_code": resp.status_code, "text": resp.text}
    body["_status_code"] = resp.status_code
    retry_after = resp.headers.get("Retry-After") if resp.headers else None
    if retry_after:
      body["_retry_after"] = retry_after
    return body

  async def poll_result(self, session_id: str) -> dict | None:
    """Poll SubAgent for result. Returns result dict or None on exhaustion.

    Transient HTTP/network errors during a single poll attempt are logged
    and treated as "not ready yet" so the loop keeps trying — they should
    not abort polling.
    """
    url = f"{self._base_url}/sessions/{session_id}/result"
    for attempt in range(1, SUBAGENT_MAX_POLL_ATTEMPTS + 1):
      loop = asyncio.get_running_loop()
      try:
        resp = await loop.run_in_executor(None, lambda: self._get_sync(url))
      except Exception:
        # Log and continue the polling loop
        await asyncio.sleep(SUBAGENT_POLL_INTERVAL)
        continue
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
    resp = requests.get(url, timeout=SUBAGENT_HTTP_TIMEOUT)
    try:
      body = resp.json()
    except Exception:
      body = {"status_code": resp.status_code, "text": resp.text}
    body["_status_code"] = resp.status_code
    return body


def _backoff_seconds(attempt: int, *, body: dict | None = None) -> float:
  """Compute backoff before retry. Honours Retry-After when present."""
  if body is not None:
    raw = body.get("_retry_after")
    if isinstance(raw, str) and raw:
      try:
        return min(SUBAGENT_SUBMIT_RETRY_MAX_BACKOFF, max(0.0, float(raw)))
      except (TypeError, ValueError):
        pass
  exp = SUBAGENT_SUBMIT_RETRY_BASE_BACKOFF * (2 ** (attempt - 1))
  return min(SUBAGENT_SUBMIT_RETRY_MAX_BACKOFF, exp)
