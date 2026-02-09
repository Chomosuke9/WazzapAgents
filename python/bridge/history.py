from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional


@dataclass
class WhatsAppMessage:
  timestamp_ms: int
  sender: str  # display name or phone
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


def format_history(messages: Iterable[WhatsAppMessage]) -> str:
  lines: list[str] = []
  for msg in messages:
    time = _fmt_time(msg.timestamp_ms)
    media_part = f"[{msg.media}]" if msg.media else ""
    text_part = msg.text or ""
    spacer = " " if media_part and text_part else ""
    lines.append(f"[{time}]{msg.sender}:{media_part}{spacer}{text_part}".strip())

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
