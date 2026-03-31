from __future__ import annotations

try:
  from .db import (
    get_permission as db_get_permission,
    permission_allows_kick,
    permission_allows_delete,
  )
  from .log import setup_logging
except ImportError:
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.db import (  # type: ignore
    get_permission as db_get_permission,
    permission_allows_kick,
    permission_allows_delete,
  )
  from bridge.log import setup_logging  # type: ignore

logger = setup_logging()


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
