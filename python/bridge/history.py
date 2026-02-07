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
  message_id: Optional[str] = None
  role: str = "user"  # "user" | "assistant"


def _fmt_time(ts_ms: int) -> str:
  dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
  return dt.strftime("%H:%M")


def format_history(messages: Iterable[WhatsAppMessage]) -> str:
  lines: list[str] = []
  for msg in messages:
    time = _fmt_time(msg.timestamp_ms)
    media_part = f"[{msg.media}]" if msg.media else ""
    text_part = msg.text or ""
    spacer = " " if media_part and text_part else ""
    lines.append(f"[{time}]{msg.sender}:{media_part}{spacer}{text_part}".strip())
  return "\n".join(lines)
