from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Dict, Optional, Tuple

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


QueueEventHandler = Callable[[str, str, int, int], Awaitable[None]]
# Signature: handler(chat_id, event_type, position, queue_size) -> awaitable


class SubAgentWebhookServer:
  # Dedup window for queue webhooks: skip a (session_id, position,
  # queue_size) tuple if we've already dispatched the same triplet in
  # the last ``_QUEUE_DEDUP_WINDOW_S`` seconds. This is a belt-and-
  # braces guard against the sub-agent double-firing after a webhook
  # retry — we do NOT want the user to see "current queue: 1" twice
  # back-to-back.
  _QUEUE_DEDUP_WINDOW_S = 5.0

  def __init__(self, tracker: SubTaskTracker, port: int | None = None) -> None:
    self._tracker = tracker
    self._port = port or SUBAGENT_WEBHOOK_PORT
    self._completion_events: Dict[str, asyncio.Event] = {}
    self._runner: web.AppRunner | None = None
    self._site: web.TCPSite | None = None
    self._queue_handler: Optional[QueueEventHandler] = None
    # session_id -> (position, queue_size, last_emit_ts)
    self._queue_last_emit: Dict[str, Tuple[int, int, float]] = {}

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

  def set_queue_handler(self, handler: Optional[QueueEventHandler]) -> None:
    """Register (or clear, with ``None``) the async handler invoked for
    every ``queued`` / ``queue_advanced`` webhook from WazzapSubAgents.

    Registered by ``main.py::handle_socket`` so the handler closes over
    the live ``ws`` connection. Cleared when the gateway disconnects so
    a stale ws is never written to.
    """
    self._queue_handler = handler

  def _dedup_queue_event(self, session_id: str, position: int, queue_size: int) -> bool:
    """Return True if this (session_id, position, queue_size) tuple is a
    dup of one emitted within the last ``_QUEUE_DEDUP_WINDOW_S`` seconds
    and should be suppressed.
    """
    now = time.time()
    prev = self._queue_last_emit.get(session_id)
    if prev is not None:
      prev_pos, prev_qs, prev_ts = prev
      if (
        prev_pos == position
        and prev_qs == queue_size
        and (now - prev_ts) < self._QUEUE_DEDUP_WINDOW_S
      ):
        return True
    self._queue_last_emit[session_id] = (position, queue_size, now)
    return False

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
      # ``reason`` is the new native-tool-call payload field; older sub-
      # agents that still emit only ``detail`` will leave it as None.
      reason = entry.get("reason")
      self._tracker.update_progress(session_id, step, detail, reason=reason)
      # Promoted from DEBUG → INFO so progress is visible at default
      # log level. The bridge previously logged at DEBUG, so operators
      # had no signal that the sub-agent was actually running unless
      # they cranked LOG_LEVEL globally.
      logger.info(
        "SubAgent progress: session=%s step=%s reason=%s detail=%s",
        session_id,
        step,
        (reason[:160] if isinstance(reason, str) else reason),
        (detail[:160] if isinstance(detail, str) else detail),
      )
      return web.json_response({"status": "ok"})

    if msg_type == "complete":
      result = data.get("result") or {}
      self._tracker.finalize(session_id, result)
      event = self._completion_events.pop(session_id, None)
      if event is not None:
        event.set()
      # Drop dedup state — once the session is finalised, any future
      # webhook with the same session_id is a real new event (extremely
      # unlikely in practice, but cheap to be tidy).
      self._queue_last_emit.pop(session_id, None)
      logger.info(
        "SubAgent complete: session=%s success=%s",
        session_id,
        result.get("success"),
      )
      return web.json_response({"status": "ok"})

    if msg_type in ("queued", "queue_advanced", "queue_status"):
      # The sub-agent is letting us know this session's position in the
      # global FIFO queue. We forward the position to the WhatsApp chat
      # via ``self._queue_handler`` (registered by main.py with a closure
      # over the live ws connection). The handler decides on the exact
      # WA wording — this layer just dedups and routes.
      try:
        position = int(data.get("position", 0) or 0)
        queue_size = int(data.get("queue_size", 0) or 0)
      except (TypeError, ValueError):
        logger.warning(
          "SubAgent queue webhook: bad position/queue_size session=%s data=%s",
          session_id,
          data,
        )
        return web.Response(status=400, text="Bad position/queue_size")

      if self._dedup_queue_event(session_id, position, queue_size):
        logger.debug(
          "SubAgent queue webhook deduped: session=%s type=%s position=%s",
          session_id,
          msg_type,
          position,
        )
        return web.json_response({"status": "deduped"})

      chat_id = self._tracker.get_chat_for_session(session_id)
      if not chat_id:
        # Session not (or no longer) tracked. Logging at INFO not WARN
        # because this is genuinely possible: the queue webhook can
        # race the bridge's own ``finalize`` call on a fast-path error.
        logger.info(
          "SubAgent queue webhook: no active task for session=%s type=%s",
          session_id,
          msg_type,
        )
        return web.json_response({"status": "no_active_task"})

      handler = self._queue_handler
      if handler is None:
        logger.info(
          "SubAgent queue webhook: no handler registered (gateway disconnected?) session=%s",
          session_id,
        )
        return web.json_response({"status": "no_handler"})

      try:
        await handler(chat_id, msg_type, position, queue_size)
      except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
          "SubAgent queue handler failed session=%s type=%s: %s",
          session_id,
          msg_type,
          exc,
        )
        return web.json_response({"status": "handler_error"}, status=500)

      logger.info(
        "SubAgent queue webhook delivered session=%s chat=%s type=%s position=%s queue_size=%s",
        session_id,
        chat_id,
        msg_type,
        position,
        queue_size,
      )
      return web.json_response({"status": "ok"})

    logger.warning("SubAgent callback: unknown type=%s", msg_type)
    return web.Response(status=400, text="Unknown type")

  async def _handle_health(self, request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})
