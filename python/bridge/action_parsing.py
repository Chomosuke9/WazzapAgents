from __future__ import annotations

import re

try:
  from .log import setup_logging
  from .stickers import resolve_sticker
  from .message_processing import (
    _normalize_context_msg_id,
    _normalize_preview_text,
    EMPTY_TARGET_TOKENS,
    SENDER_REF_RE,
  )
except ImportError:
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.log import setup_logging  # type: ignore
  from bridge.stickers import resolve_sticker  # type: ignore
  from bridge.message_processing import (  # type: ignore
    _normalize_context_msg_id,
    _normalize_preview_text,
    EMPTY_TARGET_TOKENS,
    SENDER_REF_RE,
  )

logger = setup_logging()

ACTION_LINE_RE = re.compile(r"^\[?\s*(REPLY_TO|DELETE|KICK|REACT_TO|STICKER)\s*[:=]\s*(.*?)\s*\]?$", re.IGNORECASE)
REACT_TOKEN_RE = re.compile(r"^(.+?)@(\d{6})$")


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
