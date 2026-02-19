from __future__ import annotations

import asyncio
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


def _parse_non_negative_int(raw: str | None, default: int) -> int:
  if raw is None:
    return default
  try:
    parsed = int(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed >= 0 else default


HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))
INCOMING_DEBOUNCE_SECONDS = _parse_positive_float(
  os.getenv("INCOMING_DEBOUNCE_SECONDS"), 5.0
)
INCOMING_BURST_MAX_SECONDS = _parse_positive_float(
  os.getenv("INCOMING_BURST_MAX_SECONDS"), 20.0
)
SLOW_BATCH_LOG_MS = _parse_non_negative_int(os.getenv("BRIDGE_SLOW_BATCH_LOG_MS"), 2000)
MAX_TRIGGER_BATCH_AGE_MS = _parse_non_negative_int(
  os.getenv("BRIDGE_MAX_TRIGGER_BATCH_AGE_MS"), 45000
)
REPLY_DEDUP_WINDOW_MS = _parse_non_negative_int(
  os.getenv("BRIDGE_REPLY_DEDUP_WINDOW_MS"), 120000
)
REPLY_DEDUP_MIN_CHARS = _parse_non_negative_int(
  os.getenv("BRIDGE_REPLY_DEDUP_MIN_CHARS"), 24
)
ASSISTANT_ECHO_MERGE_WINDOW_MS = _parse_non_negative_int(
  os.getenv("BRIDGE_ASSISTANT_ECHO_MERGE_WINDOW_MS"), 180000
)
logger = setup_logging()
# Accept both canonical <prompt_override> and legacy typo <prompt_overide>.
PROMPT_OVERIDE_TAG = re.compile(r"<prompt_overr?ide>([\s\S]*?)</prompt_overr?ide>", re.IGNORECASE)
ACTION_LINE_RE = re.compile(r"^\[?\s*(REPLY_TO|DELETE|KICK)\s*[:=]\s*(.*?)\s*\]?$", re.IGNORECASE)
CONTEXT_MSG_ID_RE = re.compile(r"^<?\s*(\d{6})\s*>?$")
SENDER_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,31}$")
EMPTY_TARGET_TOKENS = {"none", "null", "no", "nil", "-", ""}
REQUEST_COUNTER = count(1)
ALLOW_KICK_FLAG = "allow_kick=true"
ALLOW_DELETE_FLAG = "allow_delete=true"
ALLOW_BOTH_FLAG = "allow_kick_and_delete=true"
SYSTEM_CONTEXT_TOKEN = "system"


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
  quoted_context_id = _normalize_context_msg_id(quoted.get("contextMsgId"))
  quoted_text = _normalize_preview_text(quoted.get("text"))
  quoted_media = _infer_quoted_media(quoted)

  if sender:
    parts.append(f"from={sender}")
  if quoted_context_id:
    parts.append(f"contextId={quoted_context_id}")
  elif quoted_id:
    parts.append(f"id={quoted_id}")
  if quoted_media:
    parts.append(f"media={quoted_media}")
  if quoted_text:
    parts.append(f"text={quoted_text}")

  if not parts:
    return "reply_to:(present)"
  return f"reply_to: {' | '.join(parts)}"


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
  sender_ref = _clean_text(payload.get("senderRef")) or None
  sender_is_admin = bool(payload.get("senderIsAdmin"))
  role = "assistant" if bool(payload.get("fromMe")) else "user"
  return WhatsAppMessage(
    timestamp_ms=int(payload["timestampMs"]),
    sender=payload.get("senderName") or payload.get("senderId") or payload.get("chatId"),
    context_msg_id=context_msg_id,
    sender_ref=sender_ref,
    sender_is_admin=sender_is_admin,
    text=payload.get("text"),
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
    sender = item.get("senderName") or item.get("senderId") or item.get("chatId") or "unknown"
    sender_ref = _clean_text(item.get("senderRef")) or "unknown"
    sender_admin = "[Admin]" if bool(item.get("senderIsAdmin")) else ""
    timestamp_ms = int(item.get("timestampMs") or last.get("timestampMs") or 0)
    formatted_time = time.strftime("%H:%M", time.gmtime(max(timestamp_ms, 0) / 1000))
    text = _normalize_preview_text(item.get("text"))
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
  if not _is_context_only_payload(payload):
    return True
  return bool(payload.get("triggerLlm1"))


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
    "messagesSinceAssistantReply": _messages_since_last_assistant(
      history_before_current,
      effective_window_payloads,
    ),
    "assistantRepliesByWindow": assistant_replies_by_window,
    "humanMessagesInWindow": len(human_payloads),
  }


def _extract_prompt_override(raw_description: str | None) -> tuple[str | None, str | None]:
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
  prompt_override = "\n\n".join(prompt_blocks) if prompt_blocks else None
  return (cleaned or None), prompt_override


def _resolve_group_prompt_context(payload: dict) -> tuple[str | None, str | None]:
  raw_description = payload.get("groupDescription")
  cleaned_description, extracted_overide = _extract_prompt_override(raw_description if isinstance(raw_description, str) else None)

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


def _extract_allow_flags(prompt_override: str | None) -> set[str]:
  if not prompt_override:
    return set()
  flags: set[str] = set()
  for raw_line in prompt_override.splitlines():
    line = raw_line.strip()
    if line in {ALLOW_KICK_FLAG, ALLOW_DELETE_FLAG, ALLOW_BOTH_FLAG}:
      flags.add(line)
  return flags


def _moderation_permissions(
  prompt_override: str | None,
  *,
  bot_is_admin: bool,
  bot_is_super_admin: bool,
) -> dict:
  flags = _extract_allow_flags(prompt_override)
  admin_ok = bool(bot_is_admin or bot_is_super_admin)
  allow_both = ALLOW_BOTH_FLAG in flags
  allow_kick = admin_ok and (allow_both or ALLOW_KICK_FLAG in flags)
  allow_delete = admin_ok and (allow_both or ALLOW_DELETE_FLAG in flags)
  return {
    "allowKick": allow_kick,
    "allowDelete": allow_delete,
    "adminOk": admin_ok,
    "flags": sorted(flags),
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
        "[%s] skipped empty batch",
        chat_id,
        extra={"batch_size": len(payloads), "message_ids": [p.get("messageId") for p in payloads]},
      )
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
          "[%s] slow batch observed",
          chat_id,
          extra={
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
          "[%s] incoming_batch",
          chat_id,
          extra={
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
          logger.debug("[%s] stored context-only updates", chat_id)
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
            "[%s] skipped stale trigger batch",
            chat_id,
            extra={
              "payload_age_ms": batch_payload_age_ms,
              "max_trigger_batch_age_ms": MAX_TRIGGER_BATCH_AGE_MS,
              "trigger_batch_size": len(llm1_trigger_payloads),
            },
          )
          _log_slow_batch("stale_skip")
          return
        group_description, prompt_override = _resolve_group_prompt_context(last_payload)
        llm_context_metadata = _build_llm1_context_metadata(
          history_before_current,
          trigger_window_payloads,
        )
        llm1_payload = dict(last_payload)
        llm1_payload.update(llm_context_metadata)

        llm1_started = time.perf_counter()
        decision = await call_llm1(
          llm1_history,
          llm1_current,
          current_payload=llm1_payload,
          group_description=group_description,
          prompt_override=prompt_override,
        )
        llm1_ms = int((time.perf_counter() - llm1_started) * 1000)
        for payload in non_empty_payloads:
          _append_or_merge_history_payload(history, payload)
        if not decision.should_response:
          logger.info(
            "[%s] skipped (llm1=%s, conf=%s%%, batch=%s)",
            chat_id,
            decision.reason,
            decision.confidence,
            len(llm1_trigger_payloads),
          )
          _log_slow_batch("llm1_skip")
          return

        allowed_context_ids = _collect_context_ids(history)
        fallback_reply_to = _normalize_context_msg_id(last_payload.get("contextMsgId"))
        chat_type, bot_is_admin, bot_is_super_admin = _chat_state_from_payload(last_payload)
        llm2_payload = _merge_payload_attachments(trigger_window_payloads, last_payload)
        llm2_payload.update(llm_context_metadata)
        llm2_started = time.perf_counter()
        reply_msg = await generate_reply(
          llm2_history,
          current,
          current_payload=llm2_payload,
          group_description=group_description,
          prompt_override=prompt_override,
          chat_type=chat_type,
          bot_is_admin=bot_is_admin,
          bot_is_super_admin=bot_is_super_admin,
        )
        llm2_ms = int((time.perf_counter() - llm2_started) * 1000)
        if reply_msg is None:
          logger.warning("[%s] llm2 failed to produce reply", chat_id)
          _log_slow_batch("llm2_none")
          return
        actions = _extract_actions(
          reply_msg,
          fallback_reply_to=fallback_reply_to,
          allowed_context_ids=allowed_context_ids,
        )
        permissions = _moderation_permissions(
          prompt_override,
          bot_is_admin=bot_is_admin,
          bot_is_super_admin=bot_is_super_admin,
        )
        actions, blocked_actions = _enforce_moderation_permissions(actions, permissions)
        blocked_total = blocked_actions["kick_member"] + blocked_actions["delete_message"]
        if blocked_total > 0:
          logger.warning(
            "[%s] blocked moderation actions by backend policy",
            chat_id,
            extra={
              "blocked_actions": blocked_actions,
              "admin_ok": permissions.get("adminOk"),
              "flags": permissions.get("flags"),
              "prompt_override_present": bool(prompt_override),
            },
          )
        if not actions:
          logger.warning(
            "[%s] llm2 returned no executable action",
            chat_id,
            extra={
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
                "[%s] dropped duplicate reply",
                chat_id,
                extra={
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
            pending_send_request_chat[request_id] = chat_id
            if len(pending_send_request_chat) > 4096:
              pending_send_request_chat.pop(next(iter(pending_send_request_chat)))
            _append_history(
              history,
              WhatsAppMessage(
                timestamp_ms=int(time.time() * 1000),
                sender="LLM",
                context_msg_id=None,
                sender_ref="llm",
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
          logger.warning("[%s] unknown action type from parser: %s", chat_id, action_type)
        action_send_ms = int((time.perf_counter() - action_send_started) * 1000)
        logger.info(
          "[%s] executed actions",
          chat_id,
          extra={
            "action_counts": action_counts,
            "batch_size": len(llm1_trigger_payloads),
            "action_total": len(actions),
          },
        )
        _log_slow_batch("actions_executed", action_counts=action_counts, action_total=len(actions))
      except Exception as err:
        _log_slow_batch("handler_error")
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

      if payloads:
        await process_message_batch(payloads)
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
                  "[%s] hydrated provisional send context id from action_ack",
                  chat_id_for_request,
                  extra={
                    "request_id": request_id,
                    "context_msg_id": context_msg_id,
                  },
                )
              else:
                logger.debug(
                  "[%s] action_ack arrived but provisional send not found",
                  chat_id_for_request,
                  extra={
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


async def send_message(
  ws,
  chat_id: str,
  text: str,
  reply_to: str | None,
  *,
  request_id: str,
):
  logger.debug(
    "[%s] outbound",
    chat_id,
    extra={
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
    "[%s] outbound",
    chat_id,
    extra={
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
    "[%s] outbound",
    chat_id,
    extra={
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


def _resolve_reply_target(
  token: str | None,
  *,
  fallback_reply_to: str | None,
  allowed_context_ids: set[str],
) -> str | None:
  if token is None:
    return fallback_reply_to
  lowered = token.strip().lower()
  if lowered in EMPTY_TARGET_TOKENS:
    return None
  normalized = _normalize_context_msg_id(token)
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
  if _is_empty_target_token(token):
    return []
  parsed_targets: list[str] = []
  dedup: set[str] = set()
  parts = re.split(r"[,\s]+", token or "")
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
  if _is_empty_target_token(token):
    return []

  parsed_targets: list[dict[str, str]] = []
  dedup: set[tuple[str, str]] = set()
  segments = [segment.strip() for segment in (token or "").split(",")]
  for segment in segments:
    if not segment or "@" not in segment:
      continue
    sender_part, anchors_part = segment.split("@", 1)
    sender_ref = sender_part.strip().lower()
    if not SENDER_REF_RE.match(sender_ref):
      continue
    anchor_tokens = [anchor.strip() for anchor in anchors_part.split("|")]
    for anchor_token in anchor_tokens:
      anchor_context_msg_id = _normalize_context_msg_id(anchor_token)
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
  saw_control_line = False
  reply_declared = False
  reply_target = fallback_reply_to
  reply_lines: list[str] = []

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

  lines = text.splitlines()
  for raw_line in lines:
    stripped = raw_line.strip()
    marker = ACTION_LINE_RE.match(stripped)
    if not marker:
      if reply_declared:
        reply_lines.append(raw_line)
      else:
        orphan_lines.append(raw_line)
      continue

    saw_control_line = True
    control = marker.group(1).upper()
    value = marker.group(2).strip()
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

  flush_reply_block()

  orphan_text = "\n".join(orphan_lines).strip()
  if orphan_text:
    actions.append(
      {
        "type": "send_message",
        "text": orphan_text,
        "replyTo": fallback_reply_to,
      }
    )

  if not actions:
    if saw_control_line:
      return []
    single = text.strip()
    if single:
      actions.append(
        {
          "type": "send_message",
          "text": single,
          "replyTo": fallback_reply_to,
        }
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
