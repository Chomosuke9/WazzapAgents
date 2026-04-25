from __future__ import annotations

import asyncio
from typing import Dict

try:
  from aiohttp import web
except ImportError:  # pragma: no cover - import-time guard
  # ``aiohttp`` is a hard requirement for the SubAgent webhook server
  # (declared in requirements.txt). The fallback below keeps the import
  # itself succeeding so that callers who never instantiate the webhook
  # server (e.g. unit tests that only touch other parts of the package)
  # don't blow up — but ``SubAgentWebhookServer.start`` raises loudly
  # if it's actually missing at runtime, instead of silently degrading
  # into a 120 s polling fallback for every sub-agent task.
  web = None  # type: ignore

try:
  from ..log import setup_logging
  from .config import SUBAGENT_WEBHOOK_PORT
  from .tracker import SubTaskTracker
except ImportError:
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
  from bridge.log import setup_logging  # type: ignore
  from bridge.subagent.config import SUBAGENT_WEBHOOK_PORT  # type: ignore
  from bridge.subagent.tracker import SubTaskTracker  # type: ignore

logger = setup_logging()


class SubAgentWebhookServer:
  def __init__(self, tracker: SubTaskTracker, port: int | None = None) -> None:
    self._tracker = tracker
    self._port = port or SUBAGENT_WEBHOOK_PORT
    self._completion_events: Dict[str, asyncio.Event] = {}
    self._runner: web.AppRunner | None = None
    self._site: web.TCPSite | None = None

  async def start(self) -> None:
    if web is None:
      # Fail loud. Pre-fix this returned silently and the bridge would
      # then wait the full ``SUBAGENT_WAIT_TIMEOUT_S`` (default 120 s)
      # on every sub-agent task before falling back to polling, which
      # presented as "everything is slow" rather than a setup error.
      raise RuntimeError(
        "aiohttp is not installed but is required for the SubAgent webhook "
        "server. Install it via `pip install -r requirements.txt` (or "
        "`pip install aiohttp>=3.9.0`)."
      )
    app = web.Application()
    app.router.add_post("/subagent/callback", self._handle_callback)
    app.router.add_get("/health", self._handle_health)
    self._runner = web.AppRunner(app)
    await self._runner.setup()
    self._site = web.TCPSite(self._runner, "0.0.0.0", self._port)
    await self._site.start()
    logger.info("SubAgent webhook server started on port %s", self._port)

  async def stop(self) -> None:
    if self._site is not None:
      await self._site.stop()
      self._site = None
    if self._runner is not None:
      await self._runner.cleanup()
      self._runner = None
    logger.info("SubAgent webhook server stopped")

  def register_completion_event(self, session_id: str, event: asyncio.Event) -> None:
    self._completion_events[session_id] = event

  def unregister_completion_event(self, session_id: str) -> None:
    """Remove completion event to prevent memory leak."""
    self._completion_events.pop(session_id, None)

  async def _handle_callback(self, request: web.Request) -> web.Response:
    try:
      data = await request.json()
    except Exception:
      logger.warning("SubAgent callback: invalid JSON received")
      return web.Response(status=400, text="Invalid JSON")

    msg_type = data.get("type")
    session_id = data.get("session_id")
    if not session_id:
      logger.warning("SubAgent callback: missing session_id")
      return web.Response(status=400, text="Missing session_id")

    if msg_type == "progress":
      entry = data.get("entry") or {}
      step = entry.get("step", "unknown")
      detail = entry.get("detail", "")
      self._tracker.update_progress(session_id, step, detail)
      logger.debug(
        "SubAgent progress: session=%s step=%s",
        session_id,
        step,
      )
      return web.json_response({"status": "ok"})

    if msg_type == "complete":
      result = data.get("result") or {}
      self._tracker.finalize(session_id, result)
      event = self._completion_events.pop(session_id, None)
      if event is not None:
        event.set()
      logger.info(
        "SubAgent complete: session=%s success=%s",
        session_id,
        result.get("success"),
      )
      return web.json_response({"status": "ok"})

    logger.warning("SubAgent callback: unknown type=%s", msg_type)
    return web.Response(status=400, text="Unknown type")

  async def _handle_health(self, request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})
