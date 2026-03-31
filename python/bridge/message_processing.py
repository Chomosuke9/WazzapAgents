from __future__ import annotations

import re
import time
from collections import deque
from itertools import count
from typing import Deque

try:
  from .history import (
    WhatsAppMessage,
    assistant_name,
    assistant_sender_ref,
    format_context_time,
  )
  from .log import setup_logging
  from .config import (
    HISTORY_LIMIT,
    ASSISTANT_ECHO_MERGE_WINDOW_MS,
  )
except ImportError:
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import (  # type: ignore
    WhatsAppMessage,
    assistant_name,
    assistant_sender_ref,
    format_context_time,
  )
  from bridge.log import setup_logging  # type: ignore
  from bridge.config import (  # type: ignore
    HISTORY_LIMIT,
    ASSISTANT_ECHO_MERGE_WINDOW_MS,
  )

logger = setup_logging()

CONTEXT_MSG_ID_RE = re.compile(r"^\s*(\d{6})\s*$")
SENDER_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,31}$")
EMPTY_TARGET_TOKENS = {"none", "null", "no", "nil", "-", ""}
REQUEST_COUNTER = count(1)
SYSTEM_CONTEXT_TOKEN = "system"
MENTION_SUMMARY_MAX_ITEMS = 8


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


def _normalize_context_msg_id(value) -> str | None:
  if value is None:
    return None
  token = str(value).strip()
  if not token:
    return None
  match = CONTEXT_MSG_ID_RE.match(token)
  if not match:
    return None
  return match.group(1)


def _quoted_preview(payload: dict) -> str | None:
  quoted = _quoted_from_payload(payload)
  if not quoted:
    return None

  parts: list[str] = []
  sender = _quoted_sender(quoted)
  quoted_id = quoted.get("messageId")
  quoted_context_id = _normalize_context_msg_id(quoted.get("contextMsgId"))
  quoted_text = _normalize_preview_text(quoted.get("text"))
  quoted_media = _infer_quoted_media(quoted)
  quoted_text_is_media_stub = bool(
    quoted_text and quoted_text.startswith("<media:") and quoted_text.endswith(">")
  )

  if sender:
    parts.append(f"from={sender}")
  if quoted_context_id:
    parts.append(f"contextId={quoted_context_id}")
  elif quoted_id:
    parts.append(f"id={quoted_id}")
  if quoted_media:
    parts.append(f"quoted_media={quoted_media}")
  if quoted_text and not (quoted_media and quoted_text_is_media_stub):
    parts.append(f"quoted_text={quoted_text}")

  if not parts:
    return "reply_to:(present)"
  return f"reply_to: {' | '.join(parts)}"


def _clean_text(value) -> str:
  if isinstance(value, str):
    return value.strip()
  return ""


def _mentioned_participant_rows(payload: dict) -> list[dict]:
  rows: list[dict] = []
  mentioned_participants = payload.get("mentionedParticipants")
  if isinstance(mentioned_participants, list):
    for item in mentioned_participants:
      if not isinstance(item, dict):
        continue
      name = _clean_text(item.get("name")) or None
      sender_ref = _clean_text(item.get("senderRef")) or None
      jid = _clean_text(item.get("jid")) or None
      if not (name or sender_ref or jid):
        continue
      rows.append(
        {
          "name": name,
          "senderRef": sender_ref,
          "jid": jid,
          "isBot": bool(item.get("isBot")),
        }
      )
  if rows:
    return rows

  mentioned = payload.get("mentionedJids")
  if isinstance(mentioned, list):
    for jid in mentioned:
      token = _clean_text(jid)
      if not token:
        continue
      rows.append(
        {
          "name": None,
          "senderRef": None,
          "jid": token,
          "isBot": False,
        }
      )
  return rows


def _mention_label(row: dict) -> str:
  if bool(row.get("isBot")):
    name = _clean_text(row.get("name"))
    return f"{name} (bot)" if name else "bot (bot)"
  name = _clean_text(row.get("name"))
  sender_ref = _clean_text(row.get("senderRef"))
  jid = _clean_text(row.get("jid"))
  if name and sender_ref:
    return f"{name} ({sender_ref})"
  if name:
    return name
  if sender_ref:
    return sender_ref
  return jid


def _mention_labels(payload: dict, *, max_items: int = MENTION_SUMMARY_MAX_ITEMS) -> list[str]:
  rows = _mentioned_participant_rows(payload)
  if not rows:
    return []

  labels: list[str] = []
  seen: set[str] = set()
  for row in rows:
    label = _mention_label(row)
    if not label:
      continue
    key = label.casefold()
    if key in seen:
      continue
    seen.add(key)
    labels.append(label)
  if max_items > 0:
    return labels[:max_items]
  return labels


def _mention_number_candidates(row: dict) -> list[str]:
  jid = _clean_text(row.get("jid"))
  if not jid:
    return []
  local = jid.split("@", 1)[0].strip()
  if not local:
    return []
  candidates = [f"@{local}"]
  digits = re.sub(r"\D+", "", local)
  if digits and digits != local:
    candidates.append(f"@{digits}")
  return candidates


def _replace_mentions_in_text(base_text: str, rows: list[dict], labels: list[str]) -> tuple[str, int]:
  if not base_text or not rows or not labels:
    return base_text, 0

  rendered = base_text
  replaced = 0
  for idx, row in enumerate(rows):
    if idx >= len(labels):
      break
    replacement = f"@{labels[idx]}"
    candidates = _mention_number_candidates(row)
    applied = False
    for candidate in candidates:
      if not candidate:
        continue
      if candidate in rendered:
        rendered = rendered.replace(candidate, replacement, 1)
        replaced += 1
        applied = True
        break
    if applied:
      continue
  return rendered, replaced


def _bot_jid_from_rows(rows: list[dict]) -> str | None:
  for row in rows:
    if bool(row.get("isBot")):
      jid = _clean_text(row.get("jid"))
      if jid:
        return jid
  return None


def _ensure_bot_token_in_text(text: str, *, bot_mentioned: bool, bot_jid: str | None = None, bot_name: str = "") -> str:
  if not bot_mentioned:
    return text
  token = f"@{bot_name} (bot)" if bot_name else "@bot (bot)"
  if token in text:
    return text
  if bot_jid:
    for candidate in _mention_number_candidates({"jid": bot_jid}):
      if candidate in text:
        return text.replace(candidate, token, 1)
  stripped = text.strip()
  if stripped:
    return f"{token} {text}"
  return token


def _payload_text_with_mentions(payload: dict) -> str | None:
  base_text_raw = payload.get("text")
  base_text = base_text_raw if isinstance(base_text_raw, str) else ""
  if str(payload.get("messageType") or "").strip().lower() == "groupparticipantsupdate":
    return base_text if base_text else None

  rows = _mentioned_participant_rows(payload)
  labels = _mention_labels(payload)
  bot_jid = _bot_jid_from_rows(rows) if rows else None
  bot_mentioned = bool(payload.get("botMentioned"))
  configured_bot_name = assistant_name()
  if not rows or not labels:
    normalized = _ensure_bot_token_in_text(base_text, bot_mentioned=bot_mentioned, bot_jid=bot_jid, bot_name=configured_bot_name)
    return normalized if normalized else None

  rendered, replaced = _replace_mentions_in_text(base_text, rows, labels)
  if replaced > 0:
    normalized = _ensure_bot_token_in_text(rendered, bot_mentioned=bot_mentioned, bot_jid=bot_jid, bot_name=configured_bot_name)
    return normalized

  mention_tokens = [f"@{label}" for label in labels]
  if base_text and base_text.strip():
    normalized = f"{' '.join(mention_tokens)} {base_text}"
    return _ensure_bot_token_in_text(normalized, bot_mentioned=bot_mentioned, bot_jid=bot_jid, bot_name=configured_bot_name)
  normalized = " ".join(mention_tokens)
  return _ensure_bot_token_in_text(normalized, bot_mentioned=bot_mentioned, bot_jid=bot_jid, bot_name=configured_bot_name)


def _is_system_payload(payload: dict) -> bool:
  if not isinstance(payload, dict):
    return False
  message_type = str(payload.get("messageType") or "").strip()
  sender_id = str(payload.get("senderId") or "").strip().lower()
  if message_type == "actionLog":
    return True
  if isinstance(payload.get("groupEvent"), dict):
    return True
  if sender_id == "group-system@wazzap.local":
    return True
  return False


def _display_context_msg_id_from_payload(payload: dict) -> str:
  if _is_system_payload(payload):
    return SYSTEM_CONTEXT_TOKEN
  return _normalize_context_msg_id(payload.get("contextMsgId")) or "000000"


def _infer_media(payload: dict) -> str | None:
  atts = payload.get("attachments") or []
  if not atts:
    if payload.get("messageType") == "stickerMessage":
      return "sticker"
    return None
  return atts[0].get("kind") or atts[0].get("mime") or "media"


def _payload_to_message(payload: dict) -> WhatsAppMessage:
  quoted = _quoted_from_payload(payload)
  is_context_only = bool(payload.get("contextOnly"))
  normalized_context_msg_id = _normalize_context_msg_id(payload.get("contextMsgId"))
  context_msg_id = SYSTEM_CONTEXT_TOKEN if _is_system_payload(payload) else normalized_context_msg_id
  role = "assistant" if bool(payload.get("fromMe")) else "user"
  if role == "assistant":
    sender = assistant_name()
    sender_ref = assistant_sender_ref()
    sender_is_admin = False
  else:
    sender = payload.get("senderName") or payload.get("senderId") or payload.get("chatId")
    sender_ref = _clean_text(payload.get("senderRef")) or None
    sender_is_admin = bool(payload.get("senderIsAdmin"))
  return WhatsAppMessage(
    timestamp_ms=int(payload["timestampMs"]),
    sender=sender,
    context_msg_id=context_msg_id,
    sender_ref=sender_ref,
    sender_is_admin=sender_is_admin,
    text=_payload_text_with_mentions(payload),
    media=_infer_media(payload),
    quoted_message_id=(
      _normalize_context_msg_id(quoted.get("contextMsgId"))
      or (str(quoted.get("messageId")) if quoted.get("messageId") else None)
    ),
    quoted_sender=_quoted_sender(quoted),
    quoted_text=quoted.get("text"),
    quoted_media=_infer_quoted_media(quoted),
    message_id=None if is_context_only else (str(payload.get("messageId")) if payload.get("messageId") else None),
    role=role,
  )


def _build_burst_current(payloads: list[dict]) -> WhatsAppMessage:
  last = payloads[-1]
  if len(payloads) == 1:
    return _payload_to_message(last)

  lines: list[str] = []
  for item in payloads:
    context_msg_id = _display_context_msg_id_from_payload(item)
    if bool(item.get("fromMe")):
      sender = assistant_name()
      sender_ref = assistant_sender_ref()
      sender_admin = ""
    else:
      sender = item.get("senderName") or item.get("senderId") or item.get("chatId") or "unknown"
      sender_ref = _clean_text(item.get("senderRef")) or "unknown"
      sender_admin = "[Admin]" if bool(item.get("senderIsAdmin")) else ""
    timestamp_ms = int(item.get("timestampMs") or last.get("timestampMs") or 0)
    formatted_time = format_context_time(timestamp_ms)
    text = _normalize_preview_text(_payload_text_with_mentions(item))
    media = _infer_media(item)
    quoted = _quoted_preview(item)
    suffix = f" | {quoted}" if quoted else ""
    if text and media:
      lines.append(f"<{context_msg_id}>[{formatted_time}]{sender_admin}{sender} ({sender_ref}):[{media}] {text}{suffix}")
      continue
    if text:
      lines.append(f"<{context_msg_id}>[{formatted_time}]{sender_admin}{sender} ({sender_ref}):{text}{suffix}")
      continue
    if media:
      lines.append(f"<{context_msg_id}>[{formatted_time}]{sender_admin}{sender} ({sender_ref}):[{media}]{suffix}")
      continue
    lines.append(f"<{context_msg_id}>[{formatted_time}]{sender_admin}{sender} ({sender_ref}):(empty){suffix}")

  burst_text = (
    f"Burst messages ({len(payloads)} total, latest last):\n" + "\n".join(lines)
  )
  return WhatsAppMessage(
    timestamp_ms=int(last["timestampMs"]),
    sender=last.get("senderName") or last.get("senderId") or last.get("chatId"),
    context_msg_id=_normalize_context_msg_id(last.get("contextMsgId")),
    sender_ref=_clean_text(last.get("senderRef")) or None,
    sender_is_admin=bool(last.get("senderIsAdmin")),
    text=burst_text,
    media=_infer_media(last),
    message_id=str(last.get("messageId")) if last.get("messageId") else None,
    role="user",
  )


def _collect_context_ids(messages: list[WhatsAppMessage] | Deque[WhatsAppMessage]) -> set[str]:
  ids: set[str] = set()
  for msg in messages:
    if not msg.context_msg_id:
      continue
    ids.add(msg.context_msg_id)
  return ids


def _reply_signature(text: str | None) -> str:
  if not isinstance(text, str):
    return ""
  return " ".join(text.split()).strip().lower()


def _merge_fromme_echo_into_provisional(
  history: Deque[WhatsAppMessage],
  echo_msg: WhatsAppMessage,
) -> bool:
  if ASSISTANT_ECHO_MERGE_WINDOW_MS <= 0:
    return False
  if echo_msg.role != "assistant":
    return False

  echo_sig = _reply_signature(echo_msg.text)
  if not echo_sig:
    return False

  for idx in range(len(history) - 1, -1, -1):
    candidate = history[idx]
    if candidate.role != "assistant":
      continue

    candidate_id = candidate.message_id or ""
    if not candidate_id.startswith("local-send-"):
      continue

    age_delta = echo_msg.timestamp_ms - candidate.timestamp_ms
    if age_delta > ASSISTANT_ECHO_MERGE_WINDOW_MS:
      break

    candidate_sig = _reply_signature(candidate.text)
    if not candidate_sig or candidate_sig != echo_sig:
      continue

    candidate.timestamp_ms = echo_msg.timestamp_ms
    candidate.sender = echo_msg.sender
    candidate.context_msg_id = echo_msg.context_msg_id
    candidate.sender_ref = echo_msg.sender_ref
    candidate.sender_is_admin = echo_msg.sender_is_admin
    candidate.text = echo_msg.text
    candidate.media = echo_msg.media
    candidate.quoted_message_id = echo_msg.quoted_message_id
    candidate.quoted_sender = echo_msg.quoted_sender
    candidate.quoted_text = echo_msg.quoted_text
    candidate.quoted_media = echo_msg.quoted_media
    candidate.message_id = echo_msg.message_id
    candidate.role = echo_msg.role
    return True

  return False


def _append_or_merge_history_payload(
  history: Deque[WhatsAppMessage],
  payload: dict,
) -> None:
  msg = _payload_to_message(payload)
  if bool(payload.get("fromMe")) and _merge_fromme_echo_into_provisional(history, msg):
    return
  _append_history(history, msg)


def _extract_send_ack_context_msg_id(payload: dict) -> str | None:
  if not isinstance(payload, dict):
    return None
  result = payload.get("result")
  if not isinstance(result, dict):
    return None
  sent = result.get("sent")
  if not isinstance(sent, list):
    return None

  picked: dict | None = None
  for row in sent:
    if not isinstance(row, dict):
      continue
    if str(row.get("kind") or "").strip().lower() == "text":
      picked = row
      break
    if picked is None:
      picked = row

  if not picked:
    return None
  return _normalize_context_msg_id(picked.get("contextMsgId"))


def _hydrate_provisional_context_id_from_ack(
  history: Deque[WhatsAppMessage],
  *,
  request_id: str,
  context_msg_id: str | None,
) -> bool:
  if not request_id or not context_msg_id:
    return False
  local_message_id = f"local-send-{request_id}"
  for idx in range(len(history) - 1, -1, -1):
    candidate = history[idx]
    if candidate.role != "assistant":
      continue
    if (candidate.message_id or "") != local_message_id:
      continue
    candidate.context_msg_id = context_msg_id
    return True
  return False


def _make_request_id(action: str) -> str:
  return f"{action}-{int(time.time() * 1000)}-{next(REQUEST_COUNTER):06d}"


def _is_context_only_payload(payload: dict) -> bool:
  return bool(payload.get("contextOnly"))
