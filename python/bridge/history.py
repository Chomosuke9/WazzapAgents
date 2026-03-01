from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

DEFAULT_ASSISTANT_NAME = "LLM"
ASSISTANT_CONTEXT_SENDER_REF = "You"


def assistant_name() -> str:
  raw = os.getenv("ASSISTANT_NAME")
  if raw is None:
    return DEFAULT_ASSISTANT_NAME
  cleaned = raw.strip()
  return cleaned or DEFAULT_ASSISTANT_NAME


def assistant_sender_ref() -> str:
  return ASSISTANT_CONTEXT_SENDER_REF


@dataclass
class WhatsAppMessage:
  timestamp_ms: int
  sender: str  # display name or phone
  context_msg_id: Optional[str] = None
  sender_ref: Optional[str] = None
  sender_is_admin: bool = False
  text: Optional[str] = None
  media: Optional[str] = None  # e.g., "media", "sticker", "image", "video"
  quoted_message_id: Optional[str] = None
  quoted_sender: Optional[str] = None
  quoted_text: Optional[str] = None
  quoted_media: Optional[str] = None
  message_id: Optional[str] = None
  role: str = "user"  # "user" | "assistant"


def _context_time_utc_offset_hours() -> float | None:
  raw = os.getenv("CONTEXT_TIME_UTC_OFFSET_HOURS")
  if raw is None:
    return None
  cleaned = "".join(str(raw).split())
  if not cleaned:
    return None
  try:
    return float(cleaned)
  except (TypeError, ValueError):
    return None


def format_context_time(ts_ms: int) -> str:
  timestamp_seconds = max(ts_ms, 0) / 1000
  utc_offset_hours = _context_time_utc_offset_hours()
  if utc_offset_hours is None:
    return datetime.fromtimestamp(timestamp_seconds).strftime("%H:%M")
  dt = datetime.fromtimestamp(
    timestamp_seconds,
    tz=timezone(timedelta(hours=utc_offset_hours)),
  )
  return dt.strftime("%H:%M")


def _compact(value: Optional[str]) -> str:
  if not value:
    return ""
  return " ".join(value.split())


def _normalize_context_msg_id(value: Optional[str], *, role: str = "user") -> str:
  compact = _compact(value).lower()
  if compact in {"system", "pending"}:
    return compact
  if compact.isdigit() and len(compact) == 6:
    return compact
  if role == "assistant":
    return "pending"
  return "000000"


def _message_text(msg: WhatsAppMessage) -> str:
  media_part = f"[{msg.media}]" if msg.media else ""
  text_part = msg.text or ""
  if media_part and text_part:
    return f"{media_part} {text_part}"
  return media_part or text_part or "(empty)"


def format_history(messages: Iterable[WhatsAppMessage]) -> str:
  lines: list[str] = []
  for msg in messages:
    context_msg_id = _normalize_context_msg_id(msg.context_msg_id, role=msg.role)
    time = format_context_time(msg.timestamp_ms)
    if msg.role == "assistant":
      sender = assistant_name()
      sender_ref = assistant_sender_ref()
      admin_prefix = ""
    else:
      sender = _compact(msg.sender) or "unknown"
      sender_ref = _compact(msg.sender_ref) or "unknown"
      admin_prefix = "[Admin]" if msg.sender_is_admin else ""
    message_text = _message_text(msg)
    lines.append(f"<{context_msg_id}>[{time}]{admin_prefix}{sender} ({sender_ref}):{message_text}".rstrip())

    quote_sender = _compact(msg.quoted_sender)
    quote_text = _compact(msg.quoted_text)
    quote_media = _compact(msg.quoted_media)
    quote_id = _compact(msg.quoted_message_id)
    quote_text_is_media_stub = bool(
      quote_text and quote_text.startswith("<media:") and quote_text.endswith(">")
    )
    quote_parts: list[str] = []
    if quote_sender:
      quote_parts.append(f"from={quote_sender}")
    if quote_id:
      quote_parts.append(f"id={quote_id}")
    if quote_media:
      quote_parts.append(f"quoted_media={quote_media}")
    if quote_text and not (quote_media and quote_text_is_media_stub):
      quote_parts.append(f"quoted_text={quote_text}")
    if quote_parts:
      lines.append(f"  > reply_to: {' | '.join(quote_parts)}")
  return "\n".join(lines)
