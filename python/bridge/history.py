from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional


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


def _fmt_time(ts_ms: int) -> str:
  dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
  return dt.strftime("%H:%M")


def _compact(value: Optional[str]) -> str:
  if not value:
    return ""
  return " ".join(value.split())


def _normalize_context_msg_id(value: Optional[str]) -> str:
  compact = _compact(value)
  if compact.isdigit() and len(compact) == 6:
    return compact
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
    context_msg_id = _normalize_context_msg_id(msg.context_msg_id)
    time = _fmt_time(msg.timestamp_ms)
    sender = _compact(msg.sender) or "unknown"
    sender_ref = _compact(msg.sender_ref) or "unknown"
    admin_prefix = "[Admin]" if msg.sender_is_admin else ""
    message_text = _message_text(msg)
    lines.append(f"<{context_msg_id}>[{time}]{admin_prefix}{sender} ({sender_ref}):{message_text}".rstrip())

    quote_sender = _compact(msg.quoted_sender)
    quote_text = _compact(msg.quoted_text)
    quote_media = _compact(msg.quoted_media)
    quote_id = _compact(msg.quoted_message_id)
    quote_parts: list[str] = []
    if quote_sender:
      quote_parts.append(f"from={quote_sender}")
    if quote_id:
      quote_parts.append(f"id={quote_id}")
    if quote_media:
      quote_parts.append(f"media={quote_media}")
    if quote_text:
      quote_parts.append(f"text={quote_text}")
    if quote_parts:
      lines.append(f"  > reply_to: {' | '.join(quote_parts)}")
  return "\n".join(lines)
