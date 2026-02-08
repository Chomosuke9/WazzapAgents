from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parse_positive_int(raw: str | None, default: int) -> int:
  if raw is None:
    return default
  try:
    parsed = int(raw)
  except (TypeError, ValueError):
    return default
  return parsed if parsed > 0 else default


def llm1_media_enabled() -> bool:
  raw = os.getenv("LLM1_ENABLE_MEDIA_INPUT")
  if raw is None:
    return False
  return raw.strip().lower() in {"1", "true", "yes", "on"}


def llm2_media_enabled() -> bool:
  raw = os.getenv("LLM2_ENABLE_MEDIA_INPUT")
  if raw is None:
    return True
  return raw.strip().lower() in {"1", "true", "yes", "on"}


def media_max_items() -> int:
  return _parse_positive_int(os.getenv("LLM_MEDIA_MAX_ITEMS"), 2)


def media_max_bytes() -> int:
  return _parse_positive_int(os.getenv("LLM_MEDIA_MAX_BYTES"), 5 * 1024 * 1024)


def _resolve_local_path(path_value: Any) -> Path | None:
  if not isinstance(path_value, str) or not path_value.strip():
    return None
  path_obj = Path(path_value).expanduser()
  if path_obj.is_absolute():
    return path_obj.resolve()
  return (PROJECT_ROOT / path_obj).resolve()


def _is_visual_attachment(att: dict) -> bool:
  kind = str(att.get("kind") or "").lower()
  mime = str(att.get("mime") or "").lower()
  if kind in {"image", "sticker"}:
    return True
  return mime.startswith("image/")


def _guess_mime(att: dict, file_path: Path) -> str:
  mime = str(att.get("mime") or "").strip().lower()
  if mime.startswith("image/"):
    return mime

  guessed, _ = mimetypes.guess_type(file_path.name)
  if guessed and guessed.startswith("image/"):
    return guessed

  kind = str(att.get("kind") or "").strip().lower()
  if kind == "sticker":
    return "image/webp"
  return "image/jpeg"


def _attachment_label(att: dict) -> str:
  kind = str(att.get("kind") or "").strip().lower()
  return "sticker" if kind == "sticker" else "image"


def _placeholder_for_large_media(label: str, file_name: str) -> str:
  safe_name = file_name.replace("\n", " ").strip() or "unknown"
  return f"<{label}:too_large:{safe_name}>"


def build_visual_parts(
  payload: dict | None,
  *,
  max_items: int | None = None,
  max_bytes: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
  if not payload:
    return [], []

  attachments = payload.get("attachments") or []
  if not attachments:
    return [], []

  item_limit = max_items or media_max_items()
  size_limit = max_bytes or media_max_bytes()
  parts: list[dict[str, Any]] = []
  notes: list[str] = []
  skipped_count = 0

  for att in attachments:
    if not isinstance(att, dict):
      continue
    if not _is_visual_attachment(att):
      continue
    if len(parts) >= item_limit:
      skipped_count += 1
      continue

    path_obj = _resolve_local_path(att.get("path"))
    label = _attachment_label(att)
    file_name = str(att.get("fileName") or (path_obj.name if path_obj else "unknown"))
    if path_obj is None or not path_obj.exists() or not path_obj.is_file():
      notes.append(f"{label} skipped (missing file: {file_name})")
      skipped_count += 1
      continue

    file_size = path_obj.stat().st_size
    if file_size > size_limit:
      placeholder = _placeholder_for_large_media(label, file_name)
      notes.append(
        f"{label} placeholder: {placeholder} (size={file_size}B limit={size_limit}B)"
      )
      continue

    try:
      blob = path_obj.read_bytes()
    except Exception:
      notes.append(f"{label} skipped (read failed: {file_name})")
      skipped_count += 1
      continue

    mime = _guess_mime(att, path_obj)
    data_url = f"data:{mime};base64,{base64.b64encode(blob).decode('ascii')}"
    parts.append({"type": "image_url", "image_url": {"url": data_url}})
    notes.append(f"{label} attached: {file_name}")

  if skipped_count > 0:
    notes.append(f"{skipped_count} visual attachment(s) skipped")

  return parts, notes


def redact_multimodal_content(content: Any) -> Any:
  if isinstance(content, str):
    return content
  if not isinstance(content, list):
    return content

  redacted: list[Any] = []
  for part in content:
    if not isinstance(part, dict):
      redacted.append(part)
      continue

    if part.get("type") != "image_url":
      redacted.append(part)
      continue

    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else None
    if isinstance(url, str) and url.startswith("data:"):
      prefix = url.split(",", 1)[0]
      redacted.append(
        {
          "type": "image_url",
          "image_url": {
            "url": f"{prefix},<base64-redacted>",
          },
        }
      )
      continue

    redacted.append(part)

  return redacted
