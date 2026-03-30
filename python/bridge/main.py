from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import count
from typing import Deque, Dict, Set
from urllib.parse import urlsplit

import websockets
from dotenv import load_dotenv

try:
  from .history import (
    WhatsAppMessage,
    assistant_name,
    assistant_sender_ref,
    format_context_time,
    assistant_name_pattern,
  )
  from .log import setup_logging, set_chat_log_context, reset_chat_log_context
  from .llm1 import call_llm1, LLM1Decision
  from .llm2 import generate_reply
  from .commands import parse_command, handle_command, CommandResult
  from .db import (
    get_prompt as db_get_prompt,
    get_permission as db_get_permission,
    permission_allows_kick,
    permission_allows_delete,
    get_mode as db_get_mode,
    get_triggers as db_get_triggers,
  )
  from .dashboard import record_stat, record_user_invoke, flush_to_db, start_flush_loop
  from .stickers import resolve_sticker
except ImportError:  # allow running as `python python/bridge/main.py`
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.history import (  # type: ignore
    WhatsAppMessage,
    assistant_name,
    assistant_sender_ref,
    format_context_time,
    assistant_name_pattern,
  )
  from bridge.log import setup_logging, set_chat_log_context, reset_chat_log_context  # type: ignore
  from bridge.llm1 import call_llm1, LLM1Decision  # type: ignore
  from bridge.llm2 import generate_reply  # type: ignore
  from bridge.commands import parse_command, handle_command, CommandResult  # type: ignore
  from bridge.db import (  # type: ignore
    get_prompt as db_get_prompt,
    get_permission as db_get_permission,
    permission_allows_kick,
    permission_allows_delete,
    get_mode as db_get_mode,
    get_triggers as db_get_triggers,
  )
  from bridge.dashboard import record_stat, record_user_invoke, flush_to_db, start_flush_loop  # type: ignore
  from bridge.stickers import resolve_sticker  # type: ignore

load_dotenv()

try:
  from .config import (
    _parse_positive_float,
    _parse_non_negative_int,
    HISTORY_LIMIT,
    INCOMING_DEBOUNCE_SECONDS,
    INCOMING_BURST_MAX_SECONDS,
    SLOW_BATCH_LOG_MS,
    MAX_TRIGGER_BATCH_AGE_MS,
    REPLY_DEDUP_WINDOW_MS,
    REPLY_DEDUP_MIN_CHARS,
    ASSISTANT_ECHO_MERGE_WINDOW_MS,
  )
except ImportError:
  from bridge.config import (  # type: ignore
    _parse_positive_float,
    _parse_non_negative_int,
    HISTORY_LIMIT,
    INCOMING_DEBOUNCE_SECONDS,
    INCOMING_BURST_MAX_SECONDS,
    SLOW_BATCH_LOG_MS,
    MAX_TRIGGER_BATCH_AGE_MS,
    REPLY_DEDUP_WINDOW_MS,
    REPLY_DEDUP_MIN_CHARS,
    ASSISTANT_ECHO_MERGE_WINDOW_MS,
  )
logger = setup_logging()
ACTION_LINE_RE = re.compile(r"^\[?\s*(REPLY_TO|DELETE|KICK|REACT_TO|STICKER)\s*[:=]\s*(.*?)\s*\]?$", re.IGNORECASE)
CONTEXT_MSG_ID_RE = re.compile(r"^\s*(\d{6})\s*$")
SENDER_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,31}$")
EMPTY_TARGET_TOKENS = {"none", "null", "no", "nil", "-", ""}
REQUEST_COUNTER = count(1)
SYSTEM_CONTEXT_TOKEN = "system"
MENTION_SUMMARY_MAX_ITEMS = 8


@dataclass
class PendingChat:
  payloads: list[dict] = field(default_factory=list)
  burst_started_at: float | None = None
  last_event_at: float | None = None
  wake_event: asyncio.Event = field(default_factory=asyncio.Event)
  prefix_interrupt: asyncio.Event = field(default_factory=asyncio.Event)
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


def _chat_state_from_payload(payload: dict) -> tuple[str, bool, bool]:
  raw_chat_type = str(payload.get("chatType") or "").strip().lower()
  if raw_chat_type not in {"private", "group"}:
    raw_chat_type = "group" if bool(payload.get("isGroup")) else "private"
  return raw_chat_type, bool(payload.get("botIsAdmin")), bool(payload.get("botIsSuperAdmin"))


def _payload_timestamp_ms(payload: dict) -> int | None:
  raw = payload.get("timestampMs")
  try:
    ts = int(raw)
  except (TypeError, ValueError):
    return None
  return ts if ts > 0 else None


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


def _clean_text(value) -> str:
  if isinstance(value, str):
    return value.strip()
  return ""


def _is_context_only_payload(payload: dict) -> bool:
  return bool(payload.get("contextOnly"))


def _payload_triggers_llm1(payload: dict) -> bool:
  message_type = str(payload.get("messageType") or "").strip().lower()
  if message_type == "reactionmessage":
    return False
  if not _is_context_only_payload(payload):
    return True
  return bool(payload.get("triggerLlm1"))


def _message_matches_prefix(payload: dict, triggers: set[str]) -> bool:
  """Check if a payload matches any enabled prefix trigger."""
  if not triggers:
    return False
  if "tag" in triggers and bool(payload.get("botMentioned")):
    return True
  if "reply" in triggers and bool(payload.get("repliedToBot")):
    return True
  if "join" in triggers:
    msg_type = str(payload.get("messageType") or "").strip().lower()
    if msg_type == "groupparticipantsupdate":
      return True
  if "name" in triggers:
    text = _clean_text(payload.get("text"))
    if text and assistant_name_pattern().search(text):
      return True
  return False


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


def _payload_is_human(payload: dict) -> bool:
  # `contextOnly` non-fromMe events are synthetic context (e.g. group updates),
  # not real human-authored chat messages.
  return (not bool(payload.get("fromMe"))) and (not bool(payload.get("contextOnly")))


def _is_provisional_assistant_echo(
  history_messages: list[WhatsAppMessage],
  payload: dict,
) -> bool:
  if not bool(payload.get("fromMe")):
    return False
  if not bool(payload.get("contextOnly")):
    return False

  payload_sig = _reply_signature(payload.get("text"))
  if not payload_sig:
    return False

  payload_ts = _payload_timestamp_ms(payload)
  for msg in reversed(history_messages):
    if msg.role != "assistant":
      continue
    msg_id = msg.message_id or ""
    if not msg_id.startswith("local-send-"):
      continue
    candidate_sig = _reply_signature(msg.text)
    if not candidate_sig or candidate_sig != payload_sig:
      continue
    if payload_ts is not None and ASSISTANT_ECHO_MERGE_WINDOW_MS > 0:
      age_delta = payload_ts - msg.timestamp_ms
      if age_delta < 0 or age_delta > ASSISTANT_ECHO_MERGE_WINDOW_MS:
        continue
    return True
  return False


def _messages_since_last_assistant(
  history_messages: list[WhatsAppMessage],
  window_payloads: list[dict] | None = None,
) -> int:
  count = 0
  if window_payloads:
    for payload in reversed(window_payloads):
      if bool(payload.get("fromMe")):
        return count
      count += 1
  for msg in reversed(history_messages):
    if msg.role == "assistant":
      return count
    count += 1
  return count


def _assistant_replies_in_recent(
  history_messages: list[WhatsAppMessage],
  recent_window: int = 20,
  window_payloads: list[dict] | None = None,
) -> int:
  if recent_window <= 0:
    return 0
  combined_roles: list[str] = [msg.role for msg in history_messages]
  if window_payloads:
    combined_roles.extend(
      "assistant" if bool(payload.get("fromMe")) else "user"
      for payload in window_payloads
    )
  recent_roles = combined_roles[-recent_window:]
  return sum(1 for role in recent_roles if role == "assistant")


def _llm1_history_limit_for_metadata() -> int:
  raw = os.getenv("LLM1_HISTORY_LIMIT")
  if raw is None or not raw.strip():
    raw = os.getenv("HISTORY_LIMIT")
  try:
    parsed = int(raw) if raw is not None else HISTORY_LIMIT
  except (TypeError, ValueError):
    parsed = HISTORY_LIMIT
  if parsed > 0:
    return parsed
  return HISTORY_LIMIT if HISTORY_LIMIT > 0 else 20


def _is_group_join_action(action) -> bool:
  token = _clean_text(action).lower()
  if not token:
    return False
  if token in {"join", "add", "invite", "approve"}:
    return True
  return "join" in token


def _payload_has_explicit_join_event(payload: dict) -> bool:
  group_event = payload.get("groupEvent")
  if isinstance(group_event, dict) and _is_group_join_action(group_event.get("action")):
    return True

  message_type = str(payload.get("messageType") or "").strip().lower()
  if message_type != "groupparticipantsupdate":
    return False

  text = _clean_text(payload.get("text")).lower()
  return ("joined the group" in text) or ("new members joined the group" in text)


def _bot_name_mentioned_in_text(text: str | None, bot_name: str) -> bool:
  """Check if the bot's name appears in message text (case-insensitive, word boundary)."""
  if not text or not bot_name:
    return False
  name_lower = bot_name.strip().lower()
  if len(name_lower) < 2:
    return False
  text_lower = text.lower()
  # Use word boundary check to avoid false positives (e.g. "allow" matching "al")
  pattern = r'(?<![a-z0-9])' + re.escape(name_lower) + r'(?![a-z0-9])'
  return bool(re.search(pattern, text_lower))


def _bot_name_mentioned_in_payloads(payloads: list[dict], bot_name: str) -> bool:
  """Check if any payload in the window mentions the bot by name in text."""
  for payload in payloads:
    text = payload.get("text")
    if isinstance(text, str) and _bot_name_mentioned_in_text(text, bot_name):
      return True
    # Also check quoted text
    quoted = payload.get("quoted")
    if isinstance(quoted, dict):
      quoted_text = quoted.get("text")
      if isinstance(quoted_text, str) and _bot_name_mentioned_in_text(quoted_text, bot_name):
        return True
  return False


def _build_llm1_context_metadata(
  history_before_current: list[WhatsAppMessage],
  trigger_window_payloads: list[dict],
) -> dict:
  effective_window_payloads = [
    payload
    for payload in trigger_window_payloads
    if not _is_provisional_assistant_echo(history_before_current, payload)
  ]
  human_payloads = [payload for payload in effective_window_payloads if _payload_is_human(payload)]
  bot_mentioned_in_window = any(bool(payload.get("botMentioned")) for payload in human_payloads)
  replied_to_bot_in_window = any(bool(payload.get("repliedToBot")) for payload in human_payloads)
  bot_mention_count_in_window = sum(
    1
    for payload in human_payloads
    if bool(payload.get("botMentioned"))
  )

  # Detect bot name mentioned in text (without explicit @mention)
  configured_bot_name = assistant_name()
  bot_name_in_text = _bot_name_mentioned_in_payloads(human_payloads, configured_bot_name)

  trigger_payload = (
    effective_window_payloads[-1]
    if effective_window_payloads
    else (trigger_window_payloads[-1] if trigger_window_payloads else {})
  )
  current_has_media = bool(_infer_media(trigger_payload))
  quoted_has_media = bool(_infer_quoted_media(_quoted_from_payload(trigger_payload)))

  explicit_join_payloads = [
    payload for payload in effective_window_payloads if _payload_has_explicit_join_event(payload)
  ]
  explicit_join_participants: set[str] = set()
  for payload in explicit_join_payloads:
    group_event = payload.get("groupEvent")
    if not isinstance(group_event, dict):
      continue
    participants = group_event.get("participants")
    if not isinstance(participants, list):
      continue
    for participant in participants:
      token = _clean_text(participant)
      if token:
        explicit_join_participants.add(token.lower())

  history_limit = _llm1_history_limit_for_metadata()
  standard_windows = [20, 50, 100, 200]
  assistant_reply_windows = [window for window in standard_windows if window <= history_limit]
  if history_limit not in standard_windows:
    assistant_reply_windows.append(history_limit)
  assistant_reply_windows = sorted(set(assistant_reply_windows))
  assistant_replies_by_window = {
    str(window): _assistant_replies_in_recent(
      history_before_current,
      recent_window=window,
      window_payloads=effective_window_payloads,
    )
    for window in assistant_reply_windows
  }
  return {
    "botMentionedInWindow": bot_mentioned_in_window,
    "repliedToBotInWindow": replied_to_bot_in_window,
    "botMentionCountInWindow": bot_mention_count_in_window,
    "botNameMentionedInText": bot_name_in_text,
    "currentHasMedia": current_has_media,
    "quotedHasMedia": quoted_has_media,
    "messagesSinceAssistantReply": _messages_since_last_assistant(
      history_before_current,
      effective_window_payloads,
    ),
    "assistantRepliesByWindow": assistant_replies_by_window,
    "humanMessagesInWindow": len(human_payloads),
    "explicitJoinEventsInWindow": len(explicit_join_payloads),
    "explicitJoinParticipantsInWindow": len(explicit_join_participants),
  }


def _resolve_group_prompt_context(payload: dict) -> tuple[str | None, str | None]:
  chat_id = payload.get("chatId")
  raw_description = payload.get("groupDescription")
  description = _clean_text(raw_description) or None

  db_prompt = db_get_prompt(chat_id) if chat_id else None
  return description, db_prompt


def _moderation_permissions(
  *,
  bot_is_admin: bool,
  bot_is_super_admin: bool,
  chat_id: str | None = None,
) -> dict:
  admin_ok = bool(bot_is_admin or bot_is_super_admin)

  db_level = db_get_permission(chat_id) if chat_id else None
  allow_kick = admin_ok and permission_allows_kick(db_level)
  allow_delete = admin_ok and permission_allows_delete(db_level)
  return {
    "allowKick": allow_kick,
    "allowDelete": allow_delete,
    "adminOk": admin_ok,
    "permissionLevel": db_level,
  }


def _enforce_moderation_permissions(actions: list[dict], permissions: dict) -> tuple[list[dict], dict]:
  filtered: list[dict] = []
  blocked = {"kick_member": 0, "delete_message": 0}
  allow_kick = bool(permissions.get("allowKick"))
  allow_delete = bool(permissions.get("allowDelete"))

  for action in actions:
    action_type = action.get("type")
    if action_type == "kick_member" and not allow_kick:
      blocked["kick_member"] += 1
      continue
    if action_type == "delete_message" and not allow_delete:
      blocked["delete_message"] += 1
      continue
    filtered.append(action)

  return filtered, blocked


def _merge_payload_attachments(payloads: list[dict], base_payload: dict) -> dict:
  merged = dict(base_payload)
  merged_attachments: list[dict] = []
  seen_keys: set[str] = set()
  for payload in payloads:
    attachments = payload.get("attachments") or []
    if not isinstance(attachments, list):
      continue
    for attachment in attachments:
      if not isinstance(attachment, dict):
        continue
      path = str(attachment.get("path") or "").strip()
      kind = str(attachment.get("kind") or "").strip().lower()
      mime = str(attachment.get("mime") or "").strip().lower()
      file_name = str(attachment.get("fileName") or "").strip().lower()
      dedup_key = path or f"{kind}|{mime}|{file_name}"
      if dedup_key in seen_keys:
        continue
      seen_keys.add(dedup_key)
      merged_attachments.append(attachment)
  merged["attachments"] = merged_attachments
  return merged


async def handle_socket(ws):
  per_chat: Dict[str, Deque[WhatsAppMessage]] = defaultdict(deque)
  per_chat_lock: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
  pending_by_chat: Dict[str, PendingChat] = defaultdict(PendingChat)
  pending_send_request_chat: Dict[str, str] = {}
  recent_reply_signatures_by_chat: Dict[str, Deque[tuple[int, str]]] = defaultdict(deque)
  tasks: Set[asyncio.Task] = set()
  logger.info("Gateway connected")

  # Start dashboard flush loop
  dashboard_task = await start_flush_loop()
  tasks.add(dashboard_task)

  def _track_task(task: asyncio.Task) -> None:
    tasks.add(task)
    task.add_done_callback(tasks.discard)

  def _is_duplicate_reply(chat_id: str, text: str | None) -> bool:
    if REPLY_DEDUP_WINDOW_MS <= 0:
      return False

    signature = _reply_signature(text)
    if len(signature) < REPLY_DEDUP_MIN_CHARS:
      return False

    now_ms = int(time.time() * 1000)
    cutoff = now_ms - REPLY_DEDUP_WINDOW_MS
    items = recent_reply_signatures_by_chat[chat_id]
    while items and items[0][0] < cutoff:
      items.popleft()

    if any(prev_sig == signature for _, prev_sig in items):
      return True

    items.append((now_ms, signature))
    while len(items) > 24:
      items.popleft()
    return False

  async def process_message_batch(payloads: list[dict]):
    if not payloads:
      return

    non_empty_payloads = [payload for payload in payloads if _payload_has_meaningful_content(payload)]
    if not non_empty_payloads:
      chat_id = payloads[-1].get("chatId") if payloads else "unknown"
      logger.debug(
        "skipped empty batch",
        extra={
          "chat_id": chat_id,
          "batch_size": len(payloads),
          "message_ids": [p.get("messageId") for p in payloads],
        },
      )
      return

    # --- Slash command handling ---
    # Check each payload for slash commands. Process them, add to history, skip LLM.
    remaining_payloads: list[dict] = []
    for payload in non_empty_payloads:
      slash_cmd = payload.get("slashCommand")
      if not slash_cmd or not isinstance(slash_cmd, dict):
        remaining_payloads.append(payload)
        continue

      cmd_name = slash_cmd.get("command") or ""
      cmd_args = slash_cmd.get("args") or ""
      p_chat_id = payload.get("chatId") or "unknown"
      p_chat_type, _, _ = _chat_state_from_payload(payload)
      p_sender_is_admin = bool(payload.get("senderIsAdmin")) or bool(payload.get("senderIsOwner"))
      p_sender_jid = payload.get("senderId")

      # Add command message to history
      history = per_chat[p_chat_id]
      _append_or_merge_history_payload(history, payload)

      # Handle /reset specially: clear memory
      if cmd_name == "reset":
        result = handle_command(
          cmd_name, cmd_args,
          chat_id=p_chat_id,
          chat_type=p_chat_type,
          sender_is_admin=p_sender_is_admin,
          sender_jid=p_sender_jid,
        )
        if result and result.success:
          per_chat[p_chat_id].clear()
          logger.info("Memory cleared for chat_id=%s via /reset", p_chat_id)
        if result and result.reply:
          reply_to = _normalize_context_msg_id(payload.get("contextMsgId"))
          await send_message(ws, p_chat_id, result.reply, reply_to, request_id=_make_request_id("cmd"))
        continue

      # Handle /prompt, /permission
      result = handle_command(
        cmd_name, cmd_args,
        chat_id=p_chat_id,
        chat_type=p_chat_type,
        sender_is_admin=p_sender_is_admin,
        sender_jid=p_sender_jid,
      )
      if result is not None and result.reply:
        reply_to = _normalize_context_msg_id(payload.get("contextMsgId"))
        await send_message(ws, p_chat_id, result.reply, reply_to, request_id=_make_request_id("cmd"))
      elif result is None:
        # Command not handled here (e.g. /broadcast handled by gateway); still skip LLM
        pass
      continue

    non_empty_payloads = remaining_payloads
    if not non_empty_payloads:
      return

    context_only_payloads = [payload for payload in non_empty_payloads if _is_context_only_payload(payload)]
    trigger_indexes = [
      idx for idx, payload in enumerate(non_empty_payloads) if _payload_triggers_llm1(payload)
    ]
    llm1_trigger_payloads = [non_empty_payloads[idx] for idx in trigger_indexes]
    last_trigger_index = trigger_indexes[-1] if trigger_indexes else None
    passive_context_payloads = [
      payload for payload in context_only_payloads if not _payload_triggers_llm1(payload)
    ]

    last_payload = llm1_trigger_payloads[-1] if llm1_trigger_payloads else non_empty_payloads[-1]
    chat_id = last_payload["chatId"]
    history = per_chat[chat_id]
    lock = per_chat_lock[chat_id]
    batch_started = time.perf_counter()
    last_payload_ts = _payload_timestamp_ms(last_payload)
    lock_wait_started = time.perf_counter()
    async with lock:
      lock_wait_ms = int((time.perf_counter() - lock_wait_started) * 1000)
      llm1_ms = 0
      llm2_ms = 0
      action_send_ms = 0

      def _log_slow_batch(outcome: str, *, action_counts: dict | None = None, action_total: int = 0):
        total_ms = int((time.perf_counter() - batch_started) * 1000)
        if total_ms < SLOW_BATCH_LOG_MS and lock_wait_ms < SLOW_BATCH_LOG_MS:
          return
        payload_age_ms = None
        if last_payload_ts is not None:
          payload_age_ms = max(0, int(time.time() * 1000) - last_payload_ts)
        logger.info(
          "slow batch observed",
          extra={
            "chat_id": chat_id,
            "outcome": outcome,
            "batch_size": len(payloads),
            "non_empty_batch_size": len(non_empty_payloads),
            "llm1_trigger_batch_size": len(llm1_trigger_payloads),
            "context_only_batch_size": len(context_only_payloads),
            "passive_context_batch_size": len(passive_context_payloads),
            "lock_wait_ms": lock_wait_ms,
            "llm1_ms": llm1_ms,
            "llm2_ms": llm2_ms,
            "action_send_ms": action_send_ms,
            "total_ms": total_ms,
            "payload_age_ms": payload_age_ms,
            "history_len": len(history),
            "last_message_id": last_payload.get("messageId"),
            "last_type": last_payload.get("messageType"),
            "last_sender": last_payload.get("senderName") or last_payload.get("senderId"),
            "action_counts": action_counts,
            "action_total": action_total,
          },
        )

      try:
        logger.debug(
          "incoming_batch",
          extra={
            "chat_id": chat_id,
            "batch_size": len(payloads),
            "non_empty_batch_size": len(non_empty_payloads),
            "llm1_trigger_batch_size": len(llm1_trigger_payloads),
            "context_only_batch_size": len(context_only_payloads),
            "passive_context_batch_size": len(passive_context_payloads),
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
        if not llm1_trigger_payloads:
          for payload in non_empty_payloads:
            _append_or_merge_history_payload(history, payload)
          logger.debug("stored context-only updates", extra={"chat_id": chat_id})
          _log_slow_batch("context_only")
          return

        trigger_window_payloads = non_empty_payloads[: (last_trigger_index + 1)]
        prefix_payloads = trigger_window_payloads[:-1]
        passive_prefix_payloads = [
          payload for payload in prefix_payloads if not _payload_triggers_llm1(payload)
        ]

        history_before_current = list(history)
        current = _build_burst_current(llm1_trigger_payloads)
        llm1_history = list(history_before_current)
        # LLM1 should evaluate the pending window as a single "current" burst
        # so one trailing sticker does not overshadow earlier questions.
        llm1_current = _build_burst_current(trigger_window_payloads)
        llm2_history = list(history_before_current)
        llm2_history.extend(_payload_to_message(payload) for payload in passive_prefix_payloads)
        batch_payload_age_ms = None
        if last_payload_ts is not None:
          batch_payload_age_ms = max(0, int(time.time() * 1000) - last_payload_ts)
        if (
          MAX_TRIGGER_BATCH_AGE_MS > 0
          and batch_payload_age_ms is not None
          and batch_payload_age_ms > MAX_TRIGGER_BATCH_AGE_MS
        ):
          for payload in non_empty_payloads:
            _append_or_merge_history_payload(history, payload)
          logger.info(
            "skipped stale trigger batch",
            extra={
              "chat_id": chat_id,
              "payload_age_ms": batch_payload_age_ms,
              "max_trigger_batch_age_ms": MAX_TRIGGER_BATCH_AGE_MS,
              "trigger_batch_size": len(llm1_trigger_payloads),
            },
          )
          _log_slow_batch("stale_skip")
          return
        group_description, db_prompt = _resolve_group_prompt_context(last_payload)
        chat_type, bot_is_admin, bot_is_super_admin = _chat_state_from_payload(last_payload)
        llm_context_metadata = _build_llm1_context_metadata(
          history_before_current,
          trigger_window_payloads,
        )
        llm1_payload = dict(last_payload)
        llm1_payload.update(llm_context_metadata)

        # --- Dashboard: record messages processed ---
        for _dp in llm1_trigger_payloads:
          record_stat(chat_id, "messages_processed")
          if bool(_dp.get("botMentioned")):
            record_stat(chat_id, "bot_tags")
          _dp_text = _clean_text(_dp.get("text"))
          if _dp_text and assistant_name_pattern().search(_dp_text):
            record_stat(chat_id, "bot_name_mentions")

        # --- Mode-aware LLM1 decision ---
        chat_mode = db_get_mode(chat_id) if chat_type == "group" else "auto"
        triggers = db_get_triggers(chat_id) if chat_mode in ("prefix", "hybrid") else set()

        if chat_type == "private":
          decision = LLM1Decision(
            should_response=True,
            confidence=100,
            reason="Private chat: always respond to direct messages.",
          )
          llm1_ms = 0
          logger.info("private chat; skipping LLM1", extra={"chat_id": chat_id})
        elif chat_mode == "prefix":
          # Prefix mode: check if any trigger payload matches prefix
          prefix_matched_payloads = [p for p in llm1_trigger_payloads if _message_matches_prefix(p, triggers)]
          if not prefix_matched_payloads:
            # No prefix match — store history and skip
            for payload in non_empty_payloads:
              _append_or_merge_history_payload(history, payload)
            logger.info(
              "prefix mode: no match; skipping",
              extra={"chat_id": chat_id, "triggers": sorted(triggers), "batch_size": len(llm1_trigger_payloads)},
            )
            _log_slow_batch("prefix_no_match")
            return
          # Prefix matched — skip LLM1, go straight to LLM2
          decision = LLM1Decision(
            should_response=True,
            confidence=100,
            reason="Prefix mode: bot was explicitly invoked.",
          )
          llm1_ms = 0
          # Record invoking user for dashboard
          for _pp in prefix_matched_payloads:
            _pp_ref = _clean_text(_pp.get("senderRef"))
            _pp_name = _clean_text(_pp.get("senderName"))
            if _pp_ref:
              record_user_invoke(chat_id, _pp_ref, _pp_name)
          logger.info(
            "prefix mode: matched %d/%d payloads; skipping LLM1",
            len(prefix_matched_payloads), len(llm1_trigger_payloads),
            extra={"chat_id": chat_id, "triggers": sorted(triggers)},
          )
        elif chat_mode == "hybrid":
          # Hybrid mode: check prefix triggers first, fall back to auto (LLM1)
          prefix_matched_payloads = [p for p in llm1_trigger_payloads if _message_matches_prefix(p, triggers)]
          if prefix_matched_payloads:
            # Prefix matched in current batch — skip LLM1, go straight to LLM2
            decision = LLM1Decision(
              should_response=True,
              confidence=100,
              reason="Hybrid mode: bot was explicitly invoked (prefix trigger in batch).",
            )
            llm1_ms = 0
            for _pp in prefix_matched_payloads:
              _pp_ref = _clean_text(_pp.get("senderRef"))
              _pp_name = _clean_text(_pp.get("senderName"))
              if _pp_ref:
                record_user_invoke(chat_id, _pp_ref, _pp_name)
            logger.info(
              "hybrid mode: prefix matched %d/%d payloads; skipping LLM1",
              len(prefix_matched_payloads), len(llm1_trigger_payloads),
              extra={"chat_id": chat_id, "triggers": sorted(triggers)},
            )
          else:
            # No prefix match in batch — run LLM1 with cancellation support
            pending = pending_by_chat[chat_id]
            pending.prefix_interrupt.clear()
            llm1_started = time.perf_counter()

            llm1_task = asyncio.create_task(call_llm1(
              llm1_history,
              llm1_current,
              current_payload=llm1_payload,
              group_description=group_description,
              prompt_override=db_prompt,
            ))
            interrupt_wait = asyncio.create_task(pending.prefix_interrupt.wait())

            done, _pending_tasks = await asyncio.wait(
              {llm1_task, interrupt_wait},
              return_when=asyncio.FIRST_COMPLETED,
            )

            if interrupt_wait in done:
              # Prefix trigger arrived while LLM1 was running — cancel LLM1
              llm1_task.cancel()
              try:
                await llm1_task
              except (asyncio.CancelledError, Exception):
                pass
              llm1_ms = int((time.perf_counter() - llm1_started) * 1000)

              # Drain new prefix-trigger payloads from pending
              async with pending.lock:
                new_payloads = list(pending.payloads)
                pending.payloads.clear()
                pending.burst_started_at = None
                pending.last_event_at = None
                pending.prefix_interrupt.clear()

              if new_payloads:
                # Merge new payloads into current batch for LLM2
                non_empty_payloads.extend(new_payloads)
                new_trigger_payloads = [p for p in new_payloads if _payload_triggers_llm1(p)]
                llm1_trigger_payloads.extend(new_trigger_payloads)
                # Rebuild burst context for LLM2 with merged payloads
                trigger_window_payloads = list(non_empty_payloads)
                current = _build_burst_current(llm1_trigger_payloads)

              decision = LLM1Decision(
                should_response=True,
                confidence=100,
                reason="Hybrid mode: prefix trigger interrupted LLM1; responding immediately.",
              )
              # Record invoking users from new payloads
              for _np in new_payloads:
                if _message_matches_prefix(_np, triggers):
                  _np_ref = _clean_text(_np.get("senderRef"))
                  _np_name = _clean_text(_np.get("senderName"))
                  if _np_ref:
                    record_user_invoke(chat_id, _np_ref, _np_name)
              logger.info(
                "hybrid mode: prefix trigger interrupted LLM1 after %dms; merged %d new payloads",
                llm1_ms, len(new_payloads),
                extra={"chat_id": chat_id, "triggers": sorted(triggers)},
              )
            else:
              # LLM1 finished before any prefix interrupt
              interrupt_wait.cancel()
              try:
                await interrupt_wait
              except (asyncio.CancelledError, Exception):
                pass
              decision = llm1_task.result()
              llm1_ms = int((time.perf_counter() - llm1_started) * 1000)
              record_stat(chat_id, "llm1_calls")
              if decision.input_tokens:
                record_stat(chat_id, "llm1_input_tokens", decision.input_tokens)
              if decision.output_tokens:
                record_stat(chat_id, "llm1_output_tokens", decision.output_tokens)
              if decision.should_response:
                for _ap in llm1_trigger_payloads:
                  _ap_ref = _clean_text(_ap.get("senderRef"))
                  _ap_name = _clean_text(_ap.get("senderName"))
                  if _ap_ref:
                    record_user_invoke(chat_id, _ap_ref, _ap_name)
              logger.info(
                "hybrid mode: LLM1 completed in %dms (no prefix interrupt); should_response=%s",
                llm1_ms, decision.should_response,
                extra={"chat_id": chat_id, "confidence": decision.confidence},
              )
        else:
          llm1_started = time.perf_counter()
          decision = await call_llm1(
            llm1_history,
            llm1_current,
            current_payload=llm1_payload,
            group_description=group_description,
            prompt_override=db_prompt,
          )
          llm1_ms = int((time.perf_counter() - llm1_started) * 1000)
          record_stat(chat_id, "llm1_calls")
          if decision.input_tokens:
            record_stat(chat_id, "llm1_input_tokens", decision.input_tokens)
          if decision.output_tokens:
            record_stat(chat_id, "llm1_output_tokens", decision.output_tokens)
          # Record invoking user for auto mode too
          if decision.should_response:
            for _ap in llm1_trigger_payloads:
              _ap_ref = _clean_text(_ap.get("senderRef"))
              _ap_name = _clean_text(_ap.get("senderName"))
              if _ap_ref:
                record_user_invoke(chat_id, _ap_ref, _ap_name)

        # Send read receipt after LLM1 processes (regardless of decision)
        for _p in trigger_window_payloads:
          _msg_id = _p.get("messageId")
          _participant = _p.get("senderId") if _p.get("isGroup") else None
          await send_mark_read(ws, chat_id, _msg_id, _participant)

        for payload in non_empty_payloads:
          _append_or_merge_history_payload(history, payload)
        # Handle express decision from LLM1 (skip LLM2 entirely)
        if decision.react_expression and decision.react_context_msg_id:
          sticker_path = resolve_sticker(decision.react_expression)
          if sticker_path:
            logger.info(
              "llm1 express; sending sticker directly (skipping llm2)",
              extra={
                "chat_id": chat_id,
                "sticker_name": decision.react_expression,
                "react_context_msg_id": decision.react_context_msg_id,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "llm1_ms": llm1_ms,
              },
            )
            await send_sticker(
              ws,
              chat_id,
              sticker_path,
              decision.react_context_msg_id,
              request_id=_make_request_id("sticker"),
            )
            record_stat(chat_id, "stickers_sent")
          else:
            logger.info(
              "llm1 express; sending emoji react directly (skipping llm2)",
              extra={
                "chat_id": chat_id,
                "react_expression": decision.react_expression,
                "react_context_msg_id": decision.react_context_msg_id,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "llm1_ms": llm1_ms,
              },
            )
            await send_react_message(
              ws,
              chat_id,
              decision.react_context_msg_id,
              decision.react_expression,
              request_id=_make_request_id("react"),
            )
          _log_slow_batch("llm1_express")
          return

        if not decision.should_response:
          logger.info(
            "llm1 skip; no response sent",
            extra={"chat_id": chat_id},
          )
          _log_slow_batch("llm1_skip")
          return

        allowed_context_ids = _collect_context_ids(history)
        fallback_reply_to = _normalize_context_msg_id(last_payload.get("contextMsgId"))
        llm2_payload = _merge_payload_attachments(trigger_window_payloads, last_payload)
        llm2_payload.update(llm_context_metadata)
        llm2_payload.update(
          {
            "llm1ShouldResponse": decision.should_response,
            "llm1Confidence": decision.confidence,
            "llm1Reason": " ".join((decision.reason or "").split()),
          }
        )

        # Keep typing indicator alive while LLM2 generates (refreshes every 8s)
        llm2_started = time.perf_counter()
        async with typing_indicator(ws, chat_id):
          def _validate_llm2_result(result) -> bool:
            """Return True if the LLM2 output contains at least one usable action."""
            test_actions = _extract_actions(
              result,
              fallback_reply_to=fallback_reply_to,
              allowed_context_ids=allowed_context_ids,
            )
            return len(test_actions) > 0

          reply_msg = await generate_reply(
            llm2_history,
            current,
            current_payload=llm2_payload,
            group_description=group_description,
            prompt_override=db_prompt,
            chat_type=chat_type,
            bot_is_admin=bot_is_admin,
            bot_is_super_admin=bot_is_super_admin,
            result_validator=_validate_llm2_result,
          )

        llm2_ms = int((time.perf_counter() - llm2_started) * 1000)
        record_stat(chat_id, "llm2_calls")
        # Track LLM2 token usage if available
        if reply_msg is not None:
          _usage = getattr(reply_msg, "usage_metadata", None)
          if isinstance(_usage, dict):
            _in_tok = _usage.get("input_tokens", 0)
            _out_tok = _usage.get("output_tokens", 0)
            if _in_tok:
              record_stat(chat_id, "llm2_input_tokens", _in_tok)
            if _out_tok:
              record_stat(chat_id, "llm2_output_tokens", _out_tok)
        if reply_msg is None:
          record_stat(chat_id, "errors")
          logger.warning("llm2 failed to produce reply", extra={"chat_id": chat_id})
          _log_slow_batch("llm2_none")
          return
        actions = _extract_actions(
          reply_msg,
          fallback_reply_to=fallback_reply_to,
          allowed_context_ids=allowed_context_ids,
        )
        permissions = _moderation_permissions(
          bot_is_admin=bot_is_admin,
          bot_is_super_admin=bot_is_super_admin,
          chat_id=chat_id,
        )
        actions, blocked_actions = _enforce_moderation_permissions(actions, permissions)
        blocked_total = blocked_actions["kick_member"] + blocked_actions["delete_message"]
        if blocked_total > 0:
          logger.warning(
            "blocked moderation actions by backend policy",
            extra={
              "chat_id": chat_id,
              "blocked_actions": blocked_actions,
              "admin_ok": permissions.get("adminOk"),
            },
          )
        if not actions:
          logger.warning(
            "llm2 returned no executable action",
            extra={
              "chat_id": chat_id,
              "reply_preview": _extract_reply_text(reply_msg),
              "fallback_reply_to": fallback_reply_to,
              "blocked_actions": blocked_actions,
            },
          )
          _log_slow_batch("no_action")
          return

        action_counts: dict[str, int] = defaultdict(int)
        action_send_started = time.perf_counter()
        for action in actions:
          action_type = action.get("type")
          if action_type == "send_message":
            action_text = action.get("text") or ""
            if _is_duplicate_reply(chat_id, action_text):
              logger.info(
                "dropped duplicate reply",
                extra={
                  "chat_id": chat_id,
                  "reply_preview": _normalize_preview_text(action_text, limit=180),
                  "reply_dedup_window_ms": REPLY_DEDUP_WINDOW_MS,
                },
              )
              continue
            request_id = _make_request_id("send")
            await send_message(
              ws,
              chat_id,
              action_text,
              action.get("replyTo"),
              request_id=request_id,
            )
            record_stat(chat_id, "responses_sent")
            pending_send_request_chat[request_id] = chat_id
            if len(pending_send_request_chat) > 4096:
              pending_send_request_chat.pop(next(iter(pending_send_request_chat)))
            _append_history(
              history,
              WhatsAppMessage(
                timestamp_ms=int(time.time() * 1000),
                sender=assistant_name(),
                context_msg_id="pending",
                sender_ref=assistant_sender_ref(),
                sender_is_admin=False,
                text=action_text or None,
                media=None,
                quoted_message_id=_normalize_context_msg_id(action.get("replyTo")),
                quoted_sender=None,
                quoted_text=None,
                quoted_media=None,
                message_id=f"local-send-{request_id}",
                role="assistant",
              ),
            )
            action_counts[action_type] += 1
            continue
          if action_type == "delete_message":
            await send_delete_message(
              ws,
              chat_id,
              action.get("contextMsgId"),
              request_id=_make_request_id("delete"),
            )
            action_counts[action_type] += 1
            continue
          if action_type == "kick_member":
            await send_kick_member(
              ws,
              chat_id,
              action.get("targets") or [],
              request_id=_make_request_id("kick"),
              mode=action.get("mode") or "partial_success",
              auto_reply_anchor=bool(action.get("autoReplyAnchor", False)),
            )
            action_counts[action_type] += 1
            continue
          if action_type == "react_message":
            await send_react_message(
              ws,
              chat_id,
              action.get("contextMsgId"),
              action.get("emoji"),
              request_id=_make_request_id("react"),
            )
            action_counts[action_type] += 1
            continue
          if action_type == "send_sticker":
            sticker_name = action.get("stickerName", "")
            sticker_path = resolve_sticker(sticker_name)
            if sticker_path:
              request_id = _make_request_id("sticker")
              await send_sticker(
                ws,
                chat_id,
                sticker_path,
                action.get("replyTo"),
                request_id=request_id,
              )
              record_stat(chat_id, "stickers_sent")
              action_counts[action_type] += 1
            else:
              logger.warning(
                "sticker not found: %s",
                sticker_name,
                extra={"chat_id": chat_id},
              )
            continue
          logger.warning(
            "unknown action type from parser: %s",
            action_type,
            extra={"chat_id": chat_id},
          )
        action_send_ms = int((time.perf_counter() - action_send_started) * 1000)
        logger.info(
          "executed actions",
          extra={
            "chat_id": chat_id,
            "action_counts": action_counts,
            "batch_size": len(llm1_trigger_payloads),
            "action_total": len(actions),
          },
        )
        _log_slow_batch("actions_executed", action_counts=action_counts, action_total=len(actions))
      except Exception as err:
        _log_slow_batch("handler_error")
        logger.exception("handler error: %s", err, extra={"chat_id": chat_id})

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

        # Skip debounce for private chats and prefix/hybrid mode matches.
        _skip_debounce = False
        _last_p = pending.payloads[-1] if pending.payloads else {}
        _flush_chat_type, _, _ = _chat_state_from_payload(_last_p)
        if _flush_chat_type == "private":
          _skip_debounce = True
        elif db_get_mode(chat_id) in ("prefix", "hybrid"):
          _flush_triggers = db_get_triggers(chat_id)
          for _fp in pending.payloads:
            if _message_matches_prefix(_fp, _flush_triggers):
              _skip_debounce = True
              break

        if _skip_debounce:
          timeout_s = 0.0
        else:
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

      if payloads:
        context_payload = payloads[-1] if payloads else {}
        context_chat_type, _, _ = _chat_state_from_payload(context_payload)
        context_chat_name = _clean_text(context_payload.get("chatName")) if context_chat_type == "group" else None
        context_token = set_chat_log_context(
          chat_id=_clean_text(context_payload.get("chatId")) or None,
          chat_name=context_chat_name or None,
        )
        try:
          await process_message_batch(payloads)
        finally:
          reset_chat_log_context(context_token)
      # Keep the same worker task alive so new payloads for the same chat
      # are drained sequentially without spawning extra waiters.

  try:
    async for raw in ws:
      try:
        event = json.loads(raw)
      except json.JSONDecodeError:
        logger.warning("Dropping non-JSON payload")
        continue

      event_type = event.get("type")

      if event_type == "hello":
        logger.info("Handshake: %s", event.get("payload"))
        continue

      if event_type in {"send_ack", "action_ack"}:
        payload = event.get("payload")
        if (
          event_type == "action_ack"
          and isinstance(payload, dict)
          and str(payload.get("action") or "") == "send_message"
        ):
          request_id = _clean_text(payload.get("requestId"))
          chat_id_for_request = pending_send_request_chat.pop(request_id, None)
          if request_id and chat_id_for_request:
            context_msg_id = _extract_send_ack_context_msg_id(payload)
            if context_msg_id:
              history = per_chat[chat_id_for_request]
              lock = per_chat_lock[chat_id_for_request]
              async with lock:
                updated = _hydrate_provisional_context_id_from_ack(
                  history,
                  request_id=request_id,
                  context_msg_id=context_msg_id,
                )
              if updated:
                logger.debug(
                  "hydrated provisional send context id from action_ack",
                  extra={
                    "chat_id": chat_id_for_request,
                    "request_id": request_id,
                    "context_msg_id": context_msg_id,
                  },
                )
              else:
                logger.debug(
                  "action_ack arrived but provisional send not found",
                  extra={
                    "chat_id": chat_id_for_request,
                    "request_id": request_id,
                    "context_msg_id": context_msg_id,
                  },
                )
        logger.debug("Gateway ack: %s", event.get("payload"))
        continue

      if event_type == "error":
        logger.warning("Gateway error: %s", event.get("payload"))
        continue

      if event_type != "incoming_message":
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
        # Signal hybrid mode: if a prefix trigger arrives while LLM1 is running
        if db_get_mode(chat_id) == "hybrid":
          _hybrid_triggers = db_get_triggers(chat_id)
          if _message_matches_prefix(payload, _hybrid_triggers):
            pending.prefix_interrupt.set()
        if pending.task is None or pending.task.done():
          task = asyncio.create_task(flush_pending(chat_id))
          pending.task = task
          _track_task(task)
        else:
          pending.wake_event.set()
  except websockets.ConnectionClosed:
    logger.info("Gateway disconnected")
  finally:
    # Flush dashboard stats before shutting down
    flush_to_db()
    for task in tasks:
      task.cancel()
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)


async def send_message(
  ws,
  chat_id: str,
  text: str,
  reply_to: str | None,
  *,
  request_id: str,
):
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "send_message",
      "request_id": request_id,
      "reply_to": reply_to,
      "text_preview": text[:200],
      "text_len": len(text or ""),
    },
  )
  await ws.send(
    json.dumps(
      {
        "type": "send_message",
        "payload": {
          "requestId": request_id,
          "chatId": chat_id,
          "text": text,
          "replyTo": reply_to,
        },
      }
    )
  )


async def send_delete_message(
  ws,
  chat_id: str,
  context_msg_id: str | None,
  *,
  request_id: str,
):
  normalized_context_msg_id = _normalize_context_msg_id(context_msg_id)
  if not normalized_context_msg_id:
    return
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "delete_message",
      "request_id": request_id,
      "context_msg_id": normalized_context_msg_id,
    },
  )
  await ws.send(
    json.dumps(
      {
        "type": "delete_message",
        "payload": {
          "requestId": request_id,
          "chatId": chat_id,
          "contextMsgId": normalized_context_msg_id,
        },
      }
    )
  )


async def send_kick_member(
  ws,
  chat_id: str,
  targets: list[dict[str, str]],
  *,
  request_id: str,
  mode: str = "partial_success",
  auto_reply_anchor: bool = False,
):
  if not targets:
    return
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "kick_member",
      "request_id": request_id,
      "targets": targets,
      "mode": mode,
      "auto_reply_anchor": auto_reply_anchor,
    },
  )
  await ws.send(
    json.dumps(
      {
        "type": "kick_member",
        "payload": {
          "requestId": request_id,
          "chatId": chat_id,
          "targets": targets,
          "mode": mode,
          "autoReplyAnchor": auto_reply_anchor,
        },
      }
    )
  )


async def send_react_message(
  ws,
  chat_id: str,
  context_msg_id: str | None,
  emoji: str | None,
  *,
  request_id: str,
):
  normalized_context_msg_id = _normalize_context_msg_id(context_msg_id)
  if not normalized_context_msg_id or not emoji:
    return
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "react_message",
      "request_id": request_id,
      "context_msg_id": normalized_context_msg_id,
      "emoji": emoji,
    },
  )
  await ws.send(
    json.dumps(
      {
        "type": "react_message",
        "payload": {
          "requestId": request_id,
          "chatId": chat_id,
          "contextMsgId": normalized_context_msg_id,
          "emoji": emoji,
        },
      }
    )
  )


async def send_sticker(
  ws,
  chat_id: str,
  sticker_path: str,
  reply_to: str | None,
  *,
  request_id: str,
):
  normalized_reply_to = _normalize_context_msg_id(reply_to) if reply_to else None
  logger.debug(
    "outbound",
    extra={
      "chat_id": chat_id,
      "action": "send_sticker",
      "request_id": request_id,
      "sticker_path": sticker_path,
      "reply_to": normalized_reply_to,
    },
  )
  payload: dict = {
    "requestId": request_id,
    "chatId": chat_id,
    "attachments": [{"kind": "sticker", "path": sticker_path}],
  }
  if normalized_reply_to:
    payload["replyTo"] = normalized_reply_to
  await ws.send(
    json.dumps({"type": "send_message", "payload": payload})
  )


async def send_mark_read(
  ws,
  chat_id: str,
  message_id: str | None,
  participant: str | None = None,
):
  """Send a read receipt signal to the gateway."""
  if not message_id:
    return
  payload: dict = {"chatId": chat_id, "messageId": message_id}
  if participant:
    payload["participant"] = participant
  try:
    await ws.send(json.dumps({"type": "mark_read", "payload": payload}))
  except Exception as err:
    logger.debug("send_mark_read failed: %s", err)


async def send_typing(ws, chat_id: str, composing: bool = True):
  """Send typing presence to the gateway."""
  presence_type = "composing" if composing else "paused"
  try:
    await ws.send(
      json.dumps({"type": "send_presence", "payload": {"chatId": chat_id, "type": presence_type}})
    )
  except Exception as err:
    logger.debug("send_typing failed: %s", err)


@contextlib.asynccontextmanager
async def typing_indicator(ws, chat_id: str, interval: float = 8.0):
  """Context manager that keeps the typing indicator alive by refreshing it periodically.

  WhatsApp's typing indicator expires after ~10-15 seconds on the client side.
  This sends a fresh 'composing' presence every *interval* seconds so the
  indicator stays visible even when LLM2 takes a long time to respond.
  """

  async def _keep_alive():
    try:
      while True:
        await asyncio.sleep(interval)
        await send_typing(ws, chat_id, composing=True)
    except asyncio.CancelledError:
      pass

  await send_typing(ws, chat_id, composing=True)
  task = asyncio.create_task(_keep_alive())
  try:
    yield
  finally:
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
      await task
    await send_typing(ws, chat_id, composing=False)


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


def _is_empty_target_token(value: str | None) -> bool:
  if value is None:
    return True
  return value.strip().lower() in EMPTY_TARGET_TOKENS


def _unwrap_angle_group(value: str | None) -> str:
  return "" if value is None else str(value).strip()



def _resolve_reply_target(
  token: str | None,
  *,
  fallback_reply_to: str | None,
  allowed_context_ids: set[str],
) -> str | None:
  if token is None:
    return fallback_reply_to
  token_value = _unwrap_angle_group(token)
  if not token_value:
    return None
  lowered = token_value.lower()
  if lowered in EMPTY_TARGET_TOKENS:
    return None
  normalized = _normalize_context_msg_id(token_value)
  if not normalized:
    logger.warning("reply target ignored: invalid context id token=%r", token)
    return None
  if allowed_context_ids and normalized not in allowed_context_ids:
    logger.warning("reply target ignored: context id %s not present in allowed context ids", normalized)
    return None
  return normalized


def _resolve_delete_target(
  token: str | None,
  *,
  allowed_context_ids: set[str],
) -> str | None:
  if _is_empty_target_token(token):
    return None
  normalized = _normalize_context_msg_id(token)
  if not normalized:
    return None
  if allowed_context_ids and normalized not in allowed_context_ids:
    return None
  return normalized


def _parse_delete_targets(
  token: str | None,
  *,
  allowed_context_ids: set[str],
) -> list[str]:
  token_value = _unwrap_angle_group(token)
  if not token_value:
    return []
  if _is_empty_target_token(token_value):
    return []
  parsed_targets: list[str] = []
  dedup: set[str] = set()
  parts = re.split(r"[,\s]+", token_value)
  for part in parts:
    normalized = _resolve_delete_target(part, allowed_context_ids=allowed_context_ids)
    if not normalized:
      continue
    if normalized in dedup:
      continue
    dedup.add(normalized)
    parsed_targets.append(normalized)
  return parsed_targets


def _parse_kick_targets(
  token: str | None,
  *,
  allowed_context_ids: set[str],
) -> list[dict[str, str]]:
  token_value = _unwrap_angle_group(token)
  if not token_value:
    return []
  if _is_empty_target_token(token_value):
    return []

  parsed_targets: list[dict[str, str]] = []
  dedup: set[tuple[str, str]] = set()
  segments = [segment.strip() for segment in token_value.split(",")]
  active_sender_ref: str | None = None
  for segment in segments:
    cleaned_segment = _unwrap_angle_group(segment)
    if not cleaned_segment:
      continue

    if "@" in cleaned_segment:
      sender_part, anchors_part = cleaned_segment.split("@", 1)
      sender_ref = _unwrap_angle_group(sender_part).strip().lower()
      if not SENDER_REF_RE.match(sender_ref):
        active_sender_ref = None
        continue
      active_sender_ref = sender_ref
      anchor_tokens = [anchors_part.strip()]
    else:
      if not active_sender_ref:
        continue
      sender_ref = active_sender_ref
      anchor_tokens = [cleaned_segment]

    for anchor_token in anchor_tokens:
      anchor_context_msg_id = _normalize_context_msg_id(_unwrap_angle_group(anchor_token))
      if not anchor_context_msg_id:
        continue
      if allowed_context_ids and anchor_context_msg_id not in allowed_context_ids:
        continue
      dedup_key = (sender_ref, anchor_context_msg_id)
      if dedup_key in dedup:
        continue
      dedup.add(dedup_key)
      parsed_targets.append(
        {
          "senderRef": sender_ref,
          "anchorContextMsgId": anchor_context_msg_id,
        }
      )
  return parsed_targets


REACT_TOKEN_RE = re.compile(r"^(.+?)@(\d{6})$")


def _parse_react_context_ids(
  token: str | None,
  *,
  allowed_context_ids: set[str],
) -> list[str]:
  """Parse ``REACT_TO:<NNNNNN,NNNNNN,...>`` value into a list of context message IDs."""
  token_value = _unwrap_angle_group(token)
  if not token_value:
    return []
  if _is_empty_target_token(token_value):
    return []

  result: list[str] = []
  seen: set[str] = set()
  for segment in token_value.split(","):
    cleaned = _unwrap_angle_group(segment.strip())
    if not cleaned:
      continue
    context_msg_id = _normalize_context_msg_id(cleaned)
    if not context_msg_id:
      continue
    if allowed_context_ids and context_msg_id not in allowed_context_ids:
      continue
    if context_msg_id in seen:
      continue
    seen.add(context_msg_id)
    result.append(context_msg_id)
  return result


def _extract_actions(
  msg,
  *,
  fallback_reply_to: str | None,
  allowed_context_ids: set[str],
) -> list[dict]:
  text = _extract_reply_text(msg)
  if not text:
    return []

  actions: list[dict] = []
  orphan_lines: list[str] = []
  reply_declared = False
  reply_target = fallback_reply_to
  reply_lines: list[str] = []
  react_declared = False
  react_context_ids: list[str] = []
  sticker_declared = False
  sticker_reply_to: str | None = None

  def flush_reply_block() -> None:
    nonlocal reply_declared, reply_target, reply_lines
    if not reply_declared:
      return
    body_text = "\n".join(reply_lines).strip()
    if body_text:
      actions.append(
        {
          "type": "send_message",
          "text": body_text,
          "replyTo": reply_target,
        }
      )
    reply_declared = False
    reply_target = fallback_reply_to
    reply_lines = []

  def flush_react_block() -> None:
    nonlocal react_declared, react_context_ids
    if not react_declared:
      return
    react_declared = False
    react_context_ids = []

  def flush_sticker_block() -> None:
    nonlocal sticker_declared, sticker_reply_to
    if not sticker_declared:
      return
    sticker_declared = False
    sticker_reply_to = None

  lines = text.splitlines()
  for raw_line in lines:
    stripped = raw_line.strip()
    marker = ACTION_LINE_RE.match(stripped)
    if not marker:
      if sticker_declared and stripped:
        # Next non-empty line after STICKER: is the sticker name
        actions.append(
          {
            "type": "send_sticker",
            "stickerName": stripped,
            "replyTo": sticker_reply_to,
          }
        )
        flush_sticker_block()
      elif react_declared and stripped:
        emoji = stripped
        for ctx_id in react_context_ids:
          actions.append(
            {
              "type": "react_message",
              "contextMsgId": ctx_id,
              "emoji": emoji,
            }
          )
        flush_react_block()
      elif reply_declared:
        reply_lines.append(raw_line)
      else:
        orphan_lines.append(raw_line)
      continue

    control = marker.group(1).upper()
    value = marker.group(2).strip()

    flush_react_block()

    if control == "REPLY_TO":
      flush_reply_block()
      reply_declared = True
      reply_target = _resolve_reply_target(
        value,
        fallback_reply_to=fallback_reply_to,
        allowed_context_ids=allowed_context_ids,
      )
      continue

    if control == "DELETE":
      for target in _parse_delete_targets(
        value,
        allowed_context_ids=allowed_context_ids,
      ):
        actions.append({"type": "delete_message", "contextMsgId": target})
      continue

    if control == "KICK":
      kick_targets = _parse_kick_targets(
        value,
        allowed_context_ids=allowed_context_ids,
      )
      if kick_targets:
        actions.append(
          {
            "type": "kick_member",
            "targets": kick_targets,
            "mode": "partial_success",
            "autoReplyAnchor": True,
          }
        )
      continue

    if control == "REACT_TO":
      flush_reply_block()
      flush_sticker_block()
      ctx_ids = _parse_react_context_ids(
        value,
        allowed_context_ids=allowed_context_ids,
      )
      if ctx_ids:
        react_declared = True
        react_context_ids = ctx_ids
      continue

    if control == "STICKER":
      flush_reply_block()
      flush_react_block()
      sticker_declared = True
      sticker_reply_to = _resolve_reply_target(
        value,
        fallback_reply_to=fallback_reply_to,
        allowed_context_ids=allowed_context_ids,
      )

  flush_reply_block()
  flush_react_block()
  flush_sticker_block()

  orphan_text = "\n".join(orphan_lines).strip()
  if orphan_text:
    logger.info(
      "dropping llm2 text outside REPLY_TO block",
      extra={
        "text_preview": _normalize_preview_text(orphan_text, limit=180),
        "fallback_reply_to": fallback_reply_to,
      },
    )

  return actions


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
