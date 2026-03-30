"""Sticker catalog scanning and resolution.

Scans ``data/stickers/`` at startup for image files and provides:
- A catalog text for injection into the LLM system prompt.
- A resolver to map sticker names (without extension) to file paths.
"""
from __future__ import annotations

from pathlib import Path

try:
  from .log import setup_logging
except ImportError:
  import sys
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.log import setup_logging  # type: ignore

logger = setup_logging()

STICKER_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "stickers"
STICKER_EXTENSIONS = {".webp", ".png", ".jpg", ".jpeg", ".gif"}

# name_without_ext (lowered) → absolute path
_catalog: dict[str, str] = {}
_catalog_loaded = False


def _scan() -> None:
  global _catalog, _catalog_loaded
  _catalog_loaded = True
  _catalog = {}

  if not STICKER_DIR.is_dir():
    STICKER_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("sticker directory created: %s", STICKER_DIR)
    return

  for f in sorted(STICKER_DIR.iterdir()):
    if not f.is_file():
      continue
    if f.suffix.lower() not in STICKER_EXTENSIONS:
      continue
    name = f.stem.lower()
    _catalog[name] = str(f)

  if _catalog:
    logger.info("loaded %d sticker(s): %s", len(_catalog), list(_catalog.keys()))
  else:
    logger.info("no stickers found in %s", STICKER_DIR)


def _ensure_loaded() -> None:
  if not _catalog_loaded:
    _scan()


def sticker_catalog_text() -> str:
  """Return formatted sticker list for system prompt injection."""
  _ensure_loaded()
  if not _catalog:
    return "(no stickers available)"
  return "\n".join(f"- {name}" for name in sorted(_catalog.keys()))


def resolve_sticker(name: str) -> str | None:
  """Find sticker file by name (without extension, case-insensitive). Returns absolute path or None."""
  _ensure_loaded()
  return _catalog.get(name.strip().lower())


def sticker_names() -> list[str]:
  """Return sorted list of available sticker names."""
  _ensure_loaded()
  return sorted(_catalog.keys())
