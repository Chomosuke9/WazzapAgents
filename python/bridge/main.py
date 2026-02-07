from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Set
from urllib.parse import urlsplit

import websockets
from dotenv import load_dotenv

try:
  from .history import WhatsAppMessage
  from .log import setup_logging
  from .llm1 import call_llm1
  from .llm2 import generate_reply
except ImportError:  # allow running as `python python/bridge/main.py`
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import WhatsAppMessage  # type: ignore
  from bridge.log import setup_logging  # type: ignore
  from bridge.llm1 import call_llm1  # type: ignore
  from bridge.llm2 import generate_reply  # type: ignore

load_dotenv()

def _parse_positive_float(raw: str | None, default: float) -> float:
  if raw is None:
    return default
  try:
    parsed = float(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed > 0 else default


HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))
INCOMING_DEBOUNCE_SECONDS = _parse_positive_float(
  os.getenv("INCOMING_DEBOUNCE_SECONDS"), 5.0
)
INCOMING_BURST_MAX_SECONDS = 20.0
logger = setup_logging()


@dataclass
class PendingChat:
  payloads: list[dict] = field(default_factory=list)
  burst_started_at: float | None = None
  last_event_at: float | None = None
  wake_event: asyncio.Event = field(default_factory=asyncio.Event)
  task: asyncio.Task | None = None
  lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _append_history(buf: Deque[WhatsAppMessage], msg: WhatsAppMessage) -> None:
  buf.append(msg)
  while len(buf) > HISTORY_LIMIT:
    buf.popleft()


def _payload_to_message(payload: dict) -> WhatsAppMessage:
  return WhatsAppMessage(
    timestamp_ms=int(payload["timestampMs"]),
    sender=payload.get("senderName") or payload.get("senderId") or payload.get("chatId"),
    text=payload.get("text"),
    media=_infer_media(payload),
    message_id=str(payload.get("messageId")) if payload.get("messageId") else None,
    role="user",
  )


def _build_burst_current(payloads: list[dict]) -> WhatsAppMessage:
  last = payloads[-1]
  if len(payloads) == 1:
    return _payload_to_message(last)

  lines: list[str] = []
  for item in payloads:
    msg_id = str(item.get("messageId") or "unknown")
    sender = item.get("senderName") or item.get("senderId") or item.get("chatId") or "unknown"
    text = (item.get("text") or "").strip()
    media = _infer_media(item)
    if text and media:
      lines.append(f"- (id={msg_id}) {sender}: [{media}] {text}")
      continue
    if text:
      lines.append(f"- (id={msg_id}) {sender}: {text}")
      continue
    if media:
      lines.append(f"- (id={msg_id}) {sender}: [{media}]")
      continue
    lines.append(f"- (id={msg_id}) {sender}: (empty)")

  burst_text = (
    f"Burst messages ({len(payloads)} total, latest last):\n" + "\n".join(lines)
  )
  return WhatsAppMessage(
    timestamp_ms=int(last["timestampMs"]),
    sender=last.get("senderName") or last.get("senderId") or last.get("chatId"),
    text=burst_text,
    media=_infer_media(last),
    message_id=str(last.get("messageId")) if last.get("messageId") else None,
    role="user",
  )


def _collect_reply_candidates(history: Deque[WhatsAppMessage]) -> list[dict[str, str]]:
  candidates: list[dict[str, str]] = []
  seen_ids: set[str] = set()

  for msg in reversed(history):
    if msg.role != "user":
      continue
    message_id = (msg.message_id or "").strip()
    if not message_id or message_id in seen_ids:
      continue
    seen_ids.add(message_id)
    sender = (msg.sender or "unknown").strip() or "unknown"
    preview = " ".join((msg.text or "").split())
    if not preview:
      preview = "(no text)"
    if len(preview) > 100:
      preview = f"{preview[:97]}..."
    candidates.append(
      {
        "message_id": message_id,
        "sender": sender,
        "preview": preview,
      }
    )

  candidates.reverse()
  return candidates


async def handle_socket(ws):
  per_chat: Dict[str, Deque[WhatsAppMessage]] = defaultdict(deque)
  per_chat_lock: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
  pending_by_chat: Dict[str, PendingChat] = defaultdict(PendingChat)
  tasks: Set[asyncio.Task] = set()
  logger.info("Gateway connected")

  def _track_task(task: asyncio.Task) -> None:
    tasks.add(task)
    task.add_done_callback(tasks.discard)

  async def process_message_batch(payloads: list[dict]):
    if not payloads:
      return

    last_payload = payloads[-1]
    chat_id = last_payload["chatId"]
    history = per_chat[chat_id]
    lock = per_chat_lock[chat_id]
    async with lock:
      try:
        logger.debug(
          "[%s] incoming_batch",
          chat_id,
          extra={
            "batch_size": len(payloads),
            "message_ids": [p.get("messageId") for p in payloads],
            "last_message_id": last_payload.get("messageId"),
            "type": last_payload.get("messageType"),
            "text": last_payload.get("text"),
            "attachments": len(last_payload.get("attachments") or []),
            "quoted": bool(last_payload.get("quoted")),
            "location": bool(last_payload.get("location")),
            "history_len": len(history),
            "sender": last_payload.get("senderName") or last_payload.get("senderId"),
            "raw_payload": last_payload,
          },
        )
        burst_messages = [_payload_to_message(payload) for payload in payloads]
        current = _build_burst_current(payloads)
        llm1_history = list(history)
        llm1_current = burst_messages[-1]
        if len(burst_messages) > 1:
          # Let LLM1 see burst context as individual messages so char limit applies per message.
          llm1_history.extend(burst_messages[:-1])

        decision = await call_llm1(llm1_history, llm1_current)
        for msg in burst_messages:
          _append_history(history, msg)
        if not decision.should_response:
          logger.info(
            "[%s] skipped (llm1=%s, conf=%s%%, batch=%s)",
            chat_id,
            decision.reason,
            decision.confidence,
            len(payloads),
          )
          return

        reply_candidates = _collect_reply_candidates(history)
        reply_candidate_ids = [
          entry["message_id"] for entry in reply_candidates if entry.get("message_id")
        ]
        fallback_reply_to = str(last_payload.get("messageId")) if last_payload.get("messageId") else None
        reply_msg = await generate_reply(
          history,
          current,
          reply_candidates=reply_candidates if reply_candidates else None,
        )
        reply_choices = _extract_reply_choices(
          reply_msg,
          fallback_reply_to=fallback_reply_to,
          allowed_reply_ids=set(reply_candidate_ids),
        )
        if not reply_choices:
          logger.warning("[%s] llm2 returned empty reply", chat_id)
          return

        for reply_text, reply_to in reply_choices:
          await send_message(ws, chat_id, reply_text, reply_to)
          _append_history(
            history,
            WhatsAppMessage(
              timestamp_ms=int(last_payload["timestampMs"]),
              sender="LLM",
              text=reply_text,
              role="assistant",
            ),
          )
        logger.info(
          "[%s] replied",
          chat_id,
          extra={
            "reply_preview": reply_choices[0][0][:120],
            "batch_size": len(payloads),
            "reply_count": len(reply_choices),
          },
        )
      except Exception as err:
        logger.exception("[%s] handler error: %s", chat_id, err)

  async def flush_pending(chat_id: str):
    pending = pending_by_chat[chat_id]
    while True:
      async with pending.lock:
        if not pending.payloads:
          pending.task = None
          return

        now = time.monotonic()
        last_event_at = pending.last_event_at or now
        burst_started_at = pending.burst_started_at or now
        quiet_deadline = last_event_at + INCOMING_DEBOUNCE_SECONDS
        hard_deadline = burst_started_at + INCOMING_BURST_MAX_SECONDS
        timeout_s = max(0.0, min(quiet_deadline, hard_deadline) - now)
        pending.wake_event.clear()
        wake_event = pending.wake_event

      try:
        await asyncio.wait_for(wake_event.wait(), timeout=timeout_s)
        continue
      except asyncio.TimeoutError:
        pass

      async with pending.lock:
        payloads = list(pending.payloads)
        pending.payloads.clear()
        pending.burst_started_at = None
        pending.last_event_at = None
        pending.task = None

      if payloads:
        await process_message_batch(payloads)
      return

  try:
    async for raw in ws:
      try:
        event = json.loads(raw)
      except json.JSONDecodeError:
        logger.warning("Dropping non-JSON payload")
        continue

      if event.get("type") == "hello":
        logger.info("Handshake: %s", event.get("payload"))
        continue

      if event.get("type") != "incoming_message":
        continue

      payload = event["payload"]
      chat_id = payload.get("chatId")
      if not chat_id:
        logger.warning("Dropping incoming_message without chatId")
        continue

      pending = pending_by_chat[chat_id]
      now = time.monotonic()
      async with pending.lock:
        if pending.burst_started_at is None:
          pending.burst_started_at = now
        pending.last_event_at = now
        pending.payloads.append(payload)
        if pending.task is None or pending.task.done():
          task = asyncio.create_task(flush_pending(chat_id))
          pending.task = task
          _track_task(task)
        else:
          pending.wake_event.set()
  except websockets.ConnectionClosed:
    logger.info("Gateway disconnected")
  finally:
    for task in tasks:
      task.cancel()
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)


async def send_message(ws, chat_id: str, text: str, reply_to: str | None):
  logger.debug(
    "[%s] outbound",
    chat_id,
    extra={"reply_to": reply_to, "text_preview": text[:200], "text_len": len(text or "")},
  )
  await ws.send(
    json.dumps(
      {
        "type": "send_message",
        "payload": {
          "chatId": chat_id,
          "text": text,
          "replyTo": reply_to,
        },
      }
    )
  )


def _infer_media(payload: dict) -> str | None:
  atts = payload.get("attachments") or []
  if not atts:
    if payload.get("messageType") == "stickerMessage":
      return "sticker"
    return None
  return atts[0].get("kind") or atts[0].get("mime") or "media"


def _extract_reply_text(msg) -> str | None:
  if hasattr(msg, "content") and isinstance(msg.content, str):
    return msg.content.strip()
  if hasattr(msg, "content") and isinstance(msg.content, list):
    parts = [part for part in msg.content if isinstance(part, str)]
    return "\n".join(parts).strip() if parts else None
  return None


def _resolve_reply_target(
  token: str,
  *,
  fallback_reply_to: str | None,
  allowed_reply_ids: set[str],
) -> str | None:
  lowered = token.lower()
  if lowered in {"none", "null", "no", "nil", "-"}:
    return None
  if allowed_reply_ids and token in allowed_reply_ids:
    return token
  return fallback_reply_to


def _extract_reply_choices(
  msg,
  *,
  fallback_reply_to: str | None,
  allowed_reply_ids: set[str],
) -> list[tuple[str, str | None]]:
  text = _extract_reply_text(msg)
  if not text:
    return []

  marker = re.compile(r"^\[?\s*REPLY_TO\s*[:=]\s*([^\]\s]+)\s*\]?$", re.IGNORECASE)
  lines = text.splitlines()
  blocks: list[tuple[str, str | None]] = []
  current_target = fallback_reply_to
  current_lines: list[str] = []
  saw_marker = False

  def flush_current() -> None:
    block_text = "\n".join(current_lines).strip()
    if block_text:
      blocks.append((block_text, current_target))

  for raw_line in lines:
    m = marker.match(raw_line.strip())
    if not m:
      current_lines.append(raw_line)
      continue

    saw_marker = True
    flush_current()
    current_lines = []
    current_target = _resolve_reply_target(
      m.group(1).strip(),
      fallback_reply_to=fallback_reply_to,
      allowed_reply_ids=allowed_reply_ids,
    )

  flush_current()

  if blocks:
    return blocks
  if saw_marker:
    return []
  single = text.strip()
  if not single:
    return []
  return [(single, fallback_reply_to)]


def _parse_endpoint(url: str):
  parsed = urlsplit(url)
  host = parsed.hostname or "0.0.0.0"
  port = parsed.port or (443 if parsed.scheme == "wss" else 8080)
  return host, port


async def main():
  endpoint = os.getenv("LLM_WS_ENDPOINT", "ws://0.0.0.0:8080/ws")
  host, port = _parse_endpoint(endpoint)
  logger.info("Listening for gateway on %s (host=%s port=%s)", endpoint, host, port)
  server = await websockets.serve(
    handle_socket,
    host=host,
    port=port,
    max_size=20 * 1024 * 1024,
    ping_interval=20,
    ping_timeout=20,
  )
  await server.wait_closed()


if __name__ == "__main__":
  asyncio.run(main())
