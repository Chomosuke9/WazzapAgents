"""SQLite database for per-chat settings (prompt, permission level).

The DB file lives at ``data/bot.db`` by default (override via ``BOT_DB_PATH``).
Only low-traffic metadata is stored here – conversation history stays in RAM.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

try:
  from .log import setup_logging
except ImportError:
  import sys
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  from bridge.log import setup_logging  # type: ignore

logger = setup_logging()

_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH: Path | None = None
_LOCAL = threading.local()


def _resolve_db_path() -> Path:
  global _DB_PATH
  if _DB_PATH is not None:
    return _DB_PATH
  raw = os.getenv("BOT_DB_PATH")
  if raw and raw.strip():
    _DB_PATH = Path(raw.strip())
  else:
    data_dir = Path(os.getenv("DATA_DIR", str(_DEFAULT_DB_DIR)))
    _DB_PATH = data_dir / "bot.db"
  _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  return _DB_PATH


def _get_conn() -> sqlite3.Connection:
  """Return a thread-local connection (one per thread)."""
  conn: sqlite3.Connection | None = getattr(_LOCAL, "conn", None)
  if conn is not None:
    return conn
  db_path = _resolve_db_path()
  conn = sqlite3.connect(str(db_path), timeout=5)
  conn.execute("PRAGMA journal_mode=WAL")
  conn.execute("PRAGMA busy_timeout=3000")
  conn.row_factory = sqlite3.Row
  _LOCAL.conn = conn
  _ensure_tables(conn)
  return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
  conn.executescript("""
    CREATE TABLE IF NOT EXISTS chat_settings (
      chat_id    TEXT PRIMARY KEY,
      prompt     TEXT,
      permission INTEGER NOT NULL DEFAULT 0,
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
  """)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_prompt(chat_id: str) -> Optional[str]:
  """Return the custom prompt for *chat_id*, or ``None`` if not set."""
  conn = _get_conn()
  row = conn.execute(
    "SELECT prompt FROM chat_settings WHERE chat_id = ?", (chat_id,)
  ).fetchone()
  if row is None:
    return None
  return row["prompt"]


def set_prompt(chat_id: str, prompt: Optional[str]) -> None:
  conn = _get_conn()
  conn.execute(
    """
    INSERT INTO chat_settings (chat_id, prompt, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      prompt = excluded.prompt,
      updated_at = excluded.updated_at
    """,
    (chat_id, prompt),
  )
  conn.commit()
  logger.info("DB set_prompt chat_id=%s len=%s", chat_id, len(prompt) if prompt else 0)


def get_permission(chat_id: str) -> int:
  """Return the permission level (0-3) for *chat_id*. Default ``0``."""
  conn = _get_conn()
  row = conn.execute(
    "SELECT permission FROM chat_settings WHERE chat_id = ?", (chat_id,)
  ).fetchone()
  if row is None:
    return 0
  return int(row["permission"])


def set_permission(chat_id: str, level: int) -> None:
  clamped = max(0, min(3, int(level)))
  conn = _get_conn()
  conn.execute(
    """
    INSERT INTO chat_settings (chat_id, permission, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      permission = excluded.permission,
      updated_at = excluded.updated_at
    """,
    (chat_id, clamped),
  )
  conn.commit()
  logger.info("DB set_permission chat_id=%s level=%s", chat_id, clamped)


def clear_settings(chat_id: str) -> None:
  """Remove all stored settings for *chat_id*."""
  conn = _get_conn()
  conn.execute("DELETE FROM chat_settings WHERE chat_id = ?", (chat_id,))
  conn.commit()


def permission_description(level: int) -> str:
  """Human-readable description of a permission level."""
  mapping = {
    0: "kick and delete FORBIDDEN",
    1: "delete ALLOWED, kick FORBIDDEN",
    2: "kick ALLOWED, delete FORBIDDEN",
    3: "kick and delete ALLOWED",
  }
  return mapping.get(level, mapping[0])


def permission_allows_kick(level: int) -> bool:
  return level in (2, 3)


def permission_allows_delete(level: int) -> bool:
  return level in (1, 3)
