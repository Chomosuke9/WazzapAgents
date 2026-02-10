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
PROMPT_OVERIDE_TAG = re.compile(r"<prompt_overide>([\s\S]*?)</prompt_overide>", re.IGNORECASE)


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


def _normalize_preview_text(value: str | None, limit: int = 220) -> str:
  text = " ".join((value or "").split())
  if len(text) <= limit:
    return text
  if limit <= 3:
    return text[:limit]
  return f"{text[:limit - 3]}..."


def _quoted_from_payload(payload: dict) -> dict:
  quoted = payload.get("quoted")
  if isinstance(quoted, dict):
    return quoted
  return {}


def _infer_quoted_media(quoted: dict) -> str | None:
  q_type = str(quoted.get("type") or "").strip().lower()
  if not q_type:
    return None
  if "sticker" in q_type:
    return "sticker"
  if "image" in q_type:
    return "image"
  if "video" in q_type:
    return "video"
  if "audio" in q_type:
    return "audio"
  if "document" in q_type:
    return "document"
  return None


def _quoted_sender(quoted: dict) -> str | None:
  sender = quoted.get("senderName") or quoted.get("senderId")
  if not sender:
    return None
  return str(sender)


def _quoted_preview(payload: dict) -> str | None:
  quoted = _quoted_from_payload(payload)
  if not quoted:
    return None

  parts: list[str] = []
  sender = _quoted_sender(quoted)
  quoted_id = quoted.get("messageId")
  quoted_text = _normalize_preview_text(quoted.get("text"))
  quoted_media = _infer_quoted_media(quoted)

  if sender:
    parts.append(f"from={sender}")
  if quoted_id:
    parts.append(f"id={quoted_id}")
  if quoted_media:
    parts.append(f"media={quoted_media}")
  if quoted_text:
    parts.append(f"text={quoted_text}")

  if not parts:
    return "reply_to:(present)"
  return f"reply_to: {' | '.join(parts)}"


def _payload_to_message(payload: dict) -> WhatsAppMessage:
  quoted = _quoted_from_payload(payload)
  is_context_only = bool(payload.get("contextOnly"))
  return WhatsAppMessage(
    timestamp_ms=int(payload["timestampMs"]),
    sender=payload.get("senderName") or payload.get("senderId") or payload.get("chatId"),
    text=payload.get("text"),
    media=_infer_media(payload),
    quoted_message_id=str(quoted.get("messageId")) if quoted.get("messageId") else None,
    quoted_sender=_quoted_sender(quoted),
    quoted_text=quoted.get("text"),
    quoted_media=_infer_quoted_media(quoted),
    message_id=None if is_context_only else (str(payload.get("messageId")) if payload.get("messageId") else None),
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
    text = _normalize_preview_text(item.get("text"))
    media = _infer_media(item)
    quoted = _quoted_preview(item)
    suffix = f" | {quoted}" if quoted else ""
    if text and media:
      lines.append(f"- (id={msg_id}) {sender}: [{media}] {text}{suffix}")
      continue
    if text:
      lines.append(f"- (id={msg_id}) {sender}: {text}{suffix}")
      continue
    if media:
      lines.append(f"- (id={msg_id}) {sender}: [{media}]{suffix}")
      continue
    lines.append(f"- (id={msg_id}) {sender}: (empty){suffix}")

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
    if not preview and msg.quoted_text:
      preview = f"reply: {' '.join(msg.quoted_text.split())}"
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


def _build_reply_aliases(
  reply_candidates: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, str]]:
  prompt_candidates: list[dict[str, str]] = []
  alias_to_message_id: dict[str, str] = {}

  for index, item in enumerate(reply_candidates, start=1):
    message_id = (item.get("message_id") or "").strip()
    if not message_id:
      continue

    alias = f"{index:02d}"
    sender = item.get("sender", "unknown")
    preview = item.get("preview", "(no text)")
    prompt_candidates.append(
      {
        "message_id": alias,
        "sender": sender,
        "preview": preview,
      }
    )
    alias_to_message_id[alias] = message_id
    # Keep raw id as accepted token for backward compatibility.
    alias_to_message_id[message_id] = message_id

  return prompt_candidates, alias_to_message_id


def _clean_text(value) -> str:
  if isinstance(value, str):
    return value.strip()
  return ""


def _is_context_only_payload(payload: dict) -> bool:
  return bool(payload.get("contextOnly"))


def _payload_has_meaningful_content(payload: dict) -> bool:
  if _is_context_only_payload(payload):
    return True

  text = _clean_text(payload.get("text"))
  if text:
    return True

  attachments = payload.get("attachments") or []
  if attachments:
    return True

  if payload.get("location"):
    return True

  quoted = payload.get("quoted")
  if isinstance(quoted, dict):
    if _clean_text(quoted.get("text")):
      return True
    if quoted.get("messageId") or quoted.get("type") or quoted.get("senderName") or quoted.get("senderId"):
      return True

  return False


def _extract_prompt_overide(raw_description: str | None) -> tuple[str | None, str | None]:
  text = raw_description if isinstance(raw_description, str) else ""
  if not text.strip():
    return None, None

  prompt_blocks: list[str] = []

  def _capture(match: re.Match) -> str:
    block = _clean_text(match.group(1))
    if block:
      prompt_blocks.append(block)
    return ""

  cleaned = PROMPT_OVERIDE_TAG.sub(_capture, text)
  cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
  prompt_overide = "\n\n".join(prompt_blocks) if prompt_blocks else None
  return (cleaned or None), prompt_overide


def _resolve_group_prompt_context(payload: dict) -> tuple[str | None, str | None]:
  raw_description = payload.get("groupDescription")
  cleaned_description, extracted_overide = _extract_prompt_overide(raw_description if isinstance(raw_description, str) else None)

  payload_overide_raw = payload.get("groupPromptOveride")
  if not payload_overide_raw:
    payload_overide_raw = payload.get("groupPromptOverride")
  payload_overide = _clean_text(payload_overide_raw) or None

  if payload_overide and extracted_overide:
    if payload_overide == extracted_overide:
      merged_overide = payload_overide
    else:
      merged_overide = f"{payload_overide}\n\n{extracted_overide}"
  else:
    merged_overide = payload_overide or extracted_overide

  return cleaned_description, merged_overide


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

    non_empty_payloads = [payload for payload in payloads if _payload_has_meaningful_content(payload)]
    if not non_empty_payloads:
      chat_id = payloads[-1].get("chatId") if payloads else "unknown"
      logger.debug(
        "[%s] skipped empty batch",
        chat_id,
        extra={"batch_size": len(payloads), "message_ids": [p.get("messageId") for p in payloads]},
      )
      return

    context_only_payloads = [payload for payload in non_empty_payloads if _is_context_only_payload(payload)]
    actionable_payloads = [payload for payload in non_empty_payloads if not _is_context_only_payload(payload)]

    last_payload = actionable_payloads[-1] if actionable_payloads else non_empty_payloads[-1]
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
            "non_empty_batch_size": len(non_empty_payloads),
            "actionable_batch_size": len(actionable_payloads),
            "context_only_batch_size": len(context_only_payloads),
            "message_ids": [p.get("messageId") for p in non_empty_payloads],
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
        for payload in context_only_payloads:
          _append_history(history, _payload_to_message(payload))

        if not actionable_payloads:
          logger.debug("[%s] stored context-only updates", chat_id)
          return

        burst_messages = [_payload_to_message(payload) for payload in actionable_payloads]
        current = _build_burst_current(actionable_payloads)
        llm1_history = list(history)
        llm1_current = burst_messages[-1]
        group_description, prompt_overide = _resolve_group_prompt_context(last_payload)
        if len(burst_messages) > 1:
          # Let LLM1 see burst context as individual messages so char limit applies per message.
          llm1_history.extend(burst_messages[:-1])

        decision = await call_llm1(
          llm1_history,
          llm1_current,
          current_payload=last_payload,
          group_description=group_description,
          prompt_overide=prompt_overide,
        )
        for msg in burst_messages:
          _append_history(history, msg)
        if not decision.should_response:
          logger.info(
            "[%s] skipped (llm1=%s, conf=%s%%, batch=%s)",
            chat_id,
            decision.reason,
            decision.confidence,
            len(actionable_payloads),
          )
          return

        reply_candidates = _collect_reply_candidates(history)
        prompt_reply_candidates, reply_alias_map = _build_reply_aliases(reply_candidates)
        reply_candidate_ids = [
          entry["message_id"] for entry in reply_candidates if entry.get("message_id")
        ]
        fallback_reply_to = str(last_payload.get("messageId")) if last_payload.get("messageId") else None
        reply_msg = await generate_reply(
          history,
          current,
          reply_candidates=prompt_reply_candidates if prompt_reply_candidates else None,
          current_payload=last_payload,
          group_description=group_description,
          prompt_overide=prompt_overide,
        )
        reply_choices = _extract_reply_choices(
          reply_msg,
          fallback_reply_to=fallback_reply_to,
          allowed_reply_ids=set(reply_candidate_ids),
          reply_alias_map=reply_alias_map,
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
            "batch_size": len(actionable_payloads),
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
  reply_alias_map: dict[str, str] | None = None,
) -> str | None:
  lowered = token.lower()
  if lowered in {"none", "null", "no", "nil", "-"}:
    return None
  if reply_alias_map:
    mapped = reply_alias_map.get(token)
    if mapped:
      return mapped
  if allowed_reply_ids and token in allowed_reply_ids:
    return token
  return fallback_reply_to


def _extract_reply_choices(
  msg,
  *,
  fallback_reply_to: str | None,
  allowed_reply_ids: set[str],
  reply_alias_map: dict[str, str] | None = None,
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
      reply_alias_map=reply_alias_map,
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
