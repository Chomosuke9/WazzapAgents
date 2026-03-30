"""SQLite database for per-chat settings (prompt, permission level).

The DB file lives at ``data/bot.db`` by default (override via ``BOT_DB_PATH``).
Only low-traffic metadata is stored here – conversation history stays in RAM.

Reads go through an in-memory cache so the LLM pipeline never hits SQLite on
the hot path.  The cache is invalidated automatically when ``set_prompt`` /
``set_permission`` is called.
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

# ---------------------------------------------------------------------------
# In-memory cache  (chat_id → value)
# ---------------------------------------------------------------------------
_prompt_cache: dict[str, Optional[str]] = {}
_permission_cache: dict[str, int] = {}
_mode_cache: dict[str, str] = {}
_triggers_cache: dict[str, str] = {}
_cache_lock = threading.Lock()

VALID_MODES = {"auto", "prefix"}
DEFAULT_MODE = "auto"
VALID_TRIGGERS = {"tag", "reply", "join", "name"}
DEFAULT_TRIGGERS = "tag,reply,name"

# Sentinel to distinguish "we looked it up and it was NULL/missing" from
# "we haven't looked it up yet".
_MISSING = object()


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

    CREATE TABLE IF NOT EXISTS chat_stats (
      chat_id      TEXT NOT NULL,
      period_type  TEXT NOT NULL,
      period_key   TEXT NOT NULL,
      stat_key     TEXT NOT NULL,
      stat_value   INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (chat_id, period_type, period_key, stat_key)
    );

    CREATE TABLE IF NOT EXISTS chat_user_stats (
      chat_id      TEXT NOT NULL,
      period_type  TEXT NOT NULL,
      period_key   TEXT NOT NULL,
      sender_ref   TEXT NOT NULL,
      sender_name  TEXT NOT NULL DEFAULT '',
      invoke_count INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (chat_id, period_type, period_key, sender_ref)
    );
  """)
  # Add mode and triggers columns (may already exist)
  for col, col_type, default in [
    ("mode", "TEXT", f"'{DEFAULT_MODE}'"),
    ("triggers", "TEXT", f"'{DEFAULT_TRIGGERS}'"),
  ]:
    try:
      conn.execute(f"ALTER TABLE chat_settings ADD COLUMN {col} {col_type} NOT NULL DEFAULT {default}")
      conn.commit()
    except sqlite3.OperationalError:
      pass  # column already exists


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_prompt(chat_id: str) -> Optional[str]:
  """Return the custom prompt for *chat_id*, or ``None`` if not set."""
  with _cache_lock:
    cached = _prompt_cache.get(chat_id, _MISSING)
  if cached is not _MISSING:
    return cached  # type: ignore[return-value]

  conn = _get_conn()
  row = conn.execute(
    "SELECT prompt FROM chat_settings WHERE chat_id = ?", (chat_id,)
  ).fetchone()
  value = row["prompt"] if row is not None else None
  with _cache_lock:
    _prompt_cache[chat_id] = value
  return value


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
  with _cache_lock:
    _prompt_cache[chat_id] = prompt
  logger.info("DB set_prompt chat_id=%s len=%s", chat_id, len(prompt) if prompt else 0)


def get_permission(chat_id: str) -> int:
  """Return the permission level (0-3) for *chat_id*. Default ``0``."""
  with _cache_lock:
    cached = _permission_cache.get(chat_id, _MISSING)
  if cached is not _MISSING:
    return cached  # type: ignore[return-value]

  conn = _get_conn()
  row = conn.execute(
    "SELECT permission FROM chat_settings WHERE chat_id = ?", (chat_id,)
  ).fetchone()
  value = int(row["permission"]) if row is not None else 0
  with _cache_lock:
    _permission_cache[chat_id] = value
  return value


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
  with _cache_lock:
    _permission_cache[chat_id] = clamped
  logger.info("DB set_permission chat_id=%s level=%s", chat_id, clamped)


def clear_settings(chat_id: str) -> None:
  """Remove all stored settings for *chat_id*."""
  conn = _get_conn()
  conn.execute("DELETE FROM chat_settings WHERE chat_id = ?", (chat_id,))
  conn.commit()
  with _cache_lock:
    _prompt_cache.pop(chat_id, None)
    _permission_cache.pop(chat_id, None)


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


# ---------------------------------------------------------------------------
# Mode / Triggers
# ---------------------------------------------------------------------------

def get_mode(chat_id: str) -> str:
  """Return the chat mode ('auto' or 'prefix'). Default 'auto'."""
  with _cache_lock:
    cached = _mode_cache.get(chat_id, _MISSING)
  if cached is not _MISSING:
    return cached  # type: ignore[return-value]

  conn = _get_conn()
  row = conn.execute(
    "SELECT mode FROM chat_settings WHERE chat_id = ?", (chat_id,)
  ).fetchone()
  value = row["mode"] if row is not None else DEFAULT_MODE
  if value not in VALID_MODES:
    value = DEFAULT_MODE
  with _cache_lock:
    _mode_cache[chat_id] = value
  return value


def set_mode(chat_id: str, mode: str) -> None:
  if mode not in VALID_MODES:
    mode = DEFAULT_MODE
  conn = _get_conn()
  conn.execute(
    """
    INSERT INTO chat_settings (chat_id, mode, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      mode = excluded.mode,
      updated_at = excluded.updated_at
    """,
    (chat_id, mode),
  )
  conn.commit()
  with _cache_lock:
    _mode_cache[chat_id] = mode
  logger.info("DB set_mode chat_id=%s mode=%s", chat_id, mode)


def get_triggers(chat_id: str) -> set[str]:
  """Return the set of enabled trigger types for *chat_id*."""
  with _cache_lock:
    cached = _triggers_cache.get(chat_id, _MISSING)
  if cached is not _MISSING:
    raw = cached  # type: ignore[assignment]
  else:
    conn = _get_conn()
    row = conn.execute(
      "SELECT triggers FROM chat_settings WHERE chat_id = ?", (chat_id,)
    ).fetchone()
    raw = row["triggers"] if row is not None else DEFAULT_TRIGGERS
    with _cache_lock:
      _triggers_cache[chat_id] = raw
  return {t.strip().lower() for t in raw.split(",") if t.strip().lower() in VALID_TRIGGERS}


def set_triggers(chat_id: str, triggers: set[str]) -> None:
  valid = {t for t in triggers if t in VALID_TRIGGERS}
  raw = ",".join(sorted(valid)) if valid else ""
  conn = _get_conn()
  conn.execute(
    """
    INSERT INTO chat_settings (chat_id, triggers, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      triggers = excluded.triggers,
      updated_at = excluded.updated_at
    """,
    (chat_id, raw),
  )
  conn.commit()
  with _cache_lock:
    _triggers_cache[chat_id] = raw
  logger.info("DB set_triggers chat_id=%s triggers=%s", chat_id, raw)


# ---------------------------------------------------------------------------
# Dashboard stats persistence
# ---------------------------------------------------------------------------

def upsert_stats_batch(rows: list[tuple[str, str, str, str, int]]) -> None:
  """Batch upsert stat counters: [(chat_id, period_type, period_key, stat_key, increment), ...]."""
  if not rows:
    return
  conn = _get_conn()
  conn.executemany(
    """
    INSERT INTO chat_stats (chat_id, period_type, period_key, stat_key, stat_value)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(chat_id, period_type, period_key, stat_key) DO UPDATE SET
      stat_value = stat_value + excluded.stat_value
    """,
    rows,
  )
  conn.commit()


def upsert_user_stats_batch(rows: list[tuple[str, str, str, str, str, int]]) -> None:
  """Batch upsert user invoke counters: [(chat_id, period_type, period_key, sender_ref, sender_name, increment), ...]."""
  if not rows:
    return
  conn = _get_conn()
  conn.executemany(
    """
    INSERT INTO chat_user_stats (chat_id, period_type, period_key, sender_ref, sender_name, invoke_count)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(chat_id, period_type, period_key, sender_ref) DO UPDATE SET
      invoke_count = invoke_count + excluded.invoke_count,
      sender_name = excluded.sender_name
    """,
    rows,
  )
  conn.commit()


def get_stats(chat_id: str, period_type: str, period_key: str) -> dict[str, int]:
  """Return {stat_key: stat_value} for a given chat and period."""
  conn = _get_conn()
  rows = conn.execute(
    "SELECT stat_key, stat_value FROM chat_stats WHERE chat_id = ? AND period_type = ? AND period_key = ?",
    (chat_id, period_type, period_key),
  ).fetchall()
  return {row["stat_key"]: row["stat_value"] for row in rows}


def get_top_users(chat_id: str, period_type: str, period_key: str, limit: int = 5) -> list[tuple[str, str, int]]:
  """Return top users [(sender_ref, sender_name, invoke_count), ...] for a period."""
  conn = _get_conn()
  rows = conn.execute(
    """
    SELECT sender_ref, sender_name, invoke_count FROM chat_user_stats
    WHERE chat_id = ? AND period_type = ? AND period_key = ?
    ORDER BY invoke_count DESC LIMIT ?
    """,
    (chat_id, period_type, period_key, limit),
  ).fetchall()
  return [(row["sender_ref"], row["sender_name"], row["invoke_count"]) for row in rows]
