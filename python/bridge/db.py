"""SQLite storage split by concern.

Hard cutover layout (default under DATA_DIR):
- settings.db   : chat_settings, llm_models
- stats.db      : chat_stats, chat_user_stats
- moderation.db : chat_mutes

Legacy bot.db is read-only migration source when split DBs are empty.
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

_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent.parent / 'data'
_SETTINGS_DB_PATH: Path | None = None
_STATS_DB_PATH: Path | None = None
_MODERATION_DB_PATH: Path | None = None
_LOCAL = threading.local()

_MIGRATION_LOCK = threading.Lock()
_MIGRATION_DONE = False

# ---------------------------------------------------------------------------
# In-memory cache  (chat_id → value)
# ---------------------------------------------------------------------------
_prompt_cache: dict[str, Optional[str]] = {}
_permission_cache: dict[str, int] = {}
_mode_cache: dict[str, str] = {}
_triggers_cache: dict[str, str] = {}
# Mute cache: {chat_id: {sender_ref: {"muted_at": str, "duration_m": int, "notified": bool}}}
_mute_cache: dict[str, dict[str, dict]] = {}
_cache_lock = threading.Lock()

VALID_MODES = {'auto', 'prefix', 'hybrid'}
DEFAULT_MODE = 'prefix'
VALID_TRIGGERS = {'tag', 'reply', 'join', 'name'}
DEFAULT_TRIGGERS = 'tag,reply,name'

# Sentinel to distinguish "we looked it up and it was NULL/missing" from
# "we haven't looked it up yet".
_MISSING = object()


def _data_dir() -> Path:
  return Path(os.getenv('DATA_DIR', str(_DEFAULT_DB_DIR)))


def _env_path(*keys: str) -> str | None:
  for key in keys:
    raw = os.getenv(key)
    if raw and raw.strip():
      return raw.strip()
  return None


def _resolve_settings_db_path() -> Path:
  global _SETTINGS_DB_PATH
  if _SETTINGS_DB_PATH is not None:
    return _SETTINGS_DB_PATH
  raw = _env_path('BOT_SETTINGS_DB_PATH', 'SETTINGS_DB_PATH')
  if raw:
    _SETTINGS_DB_PATH = Path(raw)
  else:
    _SETTINGS_DB_PATH = _data_dir() / 'settings.db'
  _SETTINGS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  return _SETTINGS_DB_PATH


def _resolve_stats_db_path() -> Path:
  global _STATS_DB_PATH
  if _STATS_DB_PATH is not None:
    return _STATS_DB_PATH
  raw = _env_path('BOT_STATS_DB_PATH', 'STATS_DB_PATH')
  if raw:
    _STATS_DB_PATH = Path(raw)
  else:
    _STATS_DB_PATH = _data_dir() / 'stats.db'
  _STATS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  return _STATS_DB_PATH


def _resolve_moderation_db_path() -> Path:
  global _MODERATION_DB_PATH
  if _MODERATION_DB_PATH is not None:
    return _MODERATION_DB_PATH
  raw = _env_path('BOT_MODERATION_DB_PATH', 'MODERATION_DB_PATH')
  if raw:
    _MODERATION_DB_PATH = Path(raw)
  else:
    _MODERATION_DB_PATH = _data_dir() / 'moderation.db'
  _MODERATION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  return _MODERATION_DB_PATH


def _resolve_legacy_bot_db_path() -> Path:
  raw = _env_path('BOT_DB_PATH')
  if raw:
    return Path(raw)
  return _data_dir() / 'bot.db'


def _new_conn(db_path: Path) -> sqlite3.Connection:
  conn = sqlite3.connect(str(db_path), timeout=5)
  conn.execute('PRAGMA journal_mode=WAL')
  conn.execute('PRAGMA busy_timeout=3000')
  conn.row_factory = sqlite3.Row
  return conn


def _ensure_settings_tables(conn: sqlite3.Connection) -> None:
  conn.executescript(
    f"""
    CREATE TABLE IF NOT EXISTS chat_settings (
      chat_id    TEXT PRIMARY KEY,
      prompt     TEXT,
      permission INTEGER NOT NULL DEFAULT 0,
      mode       TEXT NOT NULL DEFAULT '{DEFAULT_MODE}',
      triggers   TEXT NOT NULL DEFAULT '{DEFAULT_TRIGGERS}',
      llm2_model TEXT,
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS llm_models (
      model_id     TEXT PRIMARY KEY,
      display_name TEXT NOT NULL,
      description  TEXT,
      is_active    INTEGER NOT NULL DEFAULT 1,
      sort_order   INTEGER NOT NULL DEFAULT 0
    );
    """
  )
  for col, col_type, default in [
    ('mode', 'TEXT', f"'{DEFAULT_MODE}'"),
    ('triggers', 'TEXT', f"'{DEFAULT_TRIGGERS}'"),
    ('llm2_model', 'TEXT', 'NULL'),
  ]:
    try:
      conn.execute(f'ALTER TABLE chat_settings ADD COLUMN {col} {col_type} DEFAULT {default}')
      conn.commit()
    except sqlite3.OperationalError:
      pass


def _ensure_stats_tables(conn: sqlite3.Connection) -> None:
  conn.executescript(
    """
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
    """
  )


def _ensure_moderation_tables(conn: sqlite3.Connection) -> None:
  conn.executescript(
    """
    CREATE TABLE IF NOT EXISTS chat_mutes (
      chat_id     TEXT NOT NULL,
      sender_ref  TEXT NOT NULL,
      muted_at    TEXT NOT NULL DEFAULT (datetime('now')),
      duration_m  INTEGER NOT NULL,
      PRIMARY KEY (chat_id, sender_ref)
    );
    """
  )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
  row = conn.execute(
    'SELECT 1 FROM sqlite_master WHERE type = ? AND name = ? LIMIT 1',
    ('table', table_name),
  ).fetchone()
  return row is not None


def _table_has_rows(conn: sqlite3.Connection, table_name: str) -> bool:
  if not _table_exists(conn, table_name):
    return False
  row = conn.execute(f'SELECT 1 FROM "{table_name}" LIMIT 1').fetchone()
  return row is not None


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
  if not _table_exists(conn, table_name):
    return set()
  rows = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
  return {str(r['name']) for r in rows}


def _migrate_from_legacy_if_needed(
  settings_conn: sqlite3.Connection,
  stats_conn: sqlite3.Connection,
  moderation_conn: sqlite3.Connection,
) -> None:
  global _MIGRATION_DONE
  if _MIGRATION_DONE:
    return
  with _MIGRATION_LOCK:
    if _MIGRATION_DONE:
      return

    legacy_path = _resolve_legacy_bot_db_path()
    if not legacy_path.exists():
      _MIGRATION_DONE = True
      return

    split_paths = {
      _resolve_settings_db_path().resolve(),
      _resolve_stats_db_path().resolve(),
      _resolve_moderation_db_path().resolve(),
    }
    if legacy_path.resolve() in split_paths:
      _MIGRATION_DONE = True
      return

    legacy_conn = _new_conn(legacy_path)
    try:
      legacy_settings_cols = _table_columns(legacy_conn, 'chat_settings')

      if (not _table_has_rows(settings_conn, 'chat_settings')) and legacy_settings_cols:
        mode_expr = 'mode' if 'mode' in legacy_settings_cols else f"'{DEFAULT_MODE}'"
        trigger_expr = 'triggers' if 'triggers' in legacy_settings_cols else f"'{DEFAULT_TRIGGERS}'"
        llm2_expr = 'llm2_model' if 'llm2_model' in legacy_settings_cols else 'NULL'
        rows = legacy_conn.execute(
          f"""
          SELECT
            chat_id,
            prompt,
            COALESCE(permission, 0) AS permission,
            {mode_expr} AS mode,
            {trigger_expr} AS triggers,
            {llm2_expr} AS llm2_model,
            COALESCE(updated_at, datetime('now')) AS updated_at
          FROM chat_settings
          """
        ).fetchall()
        settings_conn.execute('BEGIN')
        try:
          settings_conn.executemany(
            """
            INSERT INTO chat_settings (chat_id, prompt, permission, mode, triggers, llm2_model, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
              prompt = excluded.prompt,
              permission = excluded.permission,
              mode = excluded.mode,
              triggers = excluded.triggers,
              llm2_model = excluded.llm2_model,
              updated_at = excluded.updated_at
            """,
            [
              (
                r['chat_id'],
                r['prompt'],
                int(r['permission']) if r['permission'] is not None else 0,
                r['mode'] if r['mode'] else DEFAULT_MODE,
                r['triggers'] if r['triggers'] is not None else DEFAULT_TRIGGERS,
                r['llm2_model'],
                r['updated_at'],
              )
              for r in rows
            ],
          )
          settings_conn.commit()
        except Exception:
          settings_conn.rollback()
          raise
        logger.info('Migrated legacy chat_settings rows=%s', len(rows))

      if not _table_has_rows(settings_conn, 'llm_models') and _table_exists(legacy_conn, 'llm_models'):
        rows = legacy_conn.execute(
          """
          SELECT model_id, display_name, description, COALESCE(is_active, 1) AS is_active, COALESCE(sort_order, 0) AS sort_order
          FROM llm_models
          """
        ).fetchall()
        settings_conn.execute('BEGIN')
        try:
          settings_conn.executemany(
            """
            INSERT OR REPLACE INTO llm_models (model_id, display_name, description, is_active, sort_order)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(r['model_id'], r['display_name'], r['description'], int(r['is_active']), int(r['sort_order'])) for r in rows],
          )
          settings_conn.commit()
        except Exception:
          settings_conn.rollback()
          raise
        logger.info('Migrated legacy llm_models rows=%s', len(rows))

      if not _table_has_rows(stats_conn, 'chat_stats') and _table_exists(legacy_conn, 'chat_stats'):
        rows = legacy_conn.execute(
          'SELECT chat_id, period_type, period_key, stat_key, stat_value FROM chat_stats'
        ).fetchall()
        stats_conn.execute('BEGIN')
        try:
          stats_conn.executemany(
            """
            INSERT OR REPLACE INTO chat_stats (chat_id, period_type, period_key, stat_key, stat_value)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(r['chat_id'], r['period_type'], r['period_key'], r['stat_key'], int(r['stat_value'])) for r in rows],
          )
          stats_conn.commit()
        except Exception:
          stats_conn.rollback()
          raise
        logger.info('Migrated legacy chat_stats rows=%s', len(rows))

      if not _table_has_rows(stats_conn, 'chat_user_stats') and _table_exists(legacy_conn, 'chat_user_stats'):
        rows = legacy_conn.execute(
          'SELECT chat_id, period_type, period_key, sender_ref, sender_name, invoke_count FROM chat_user_stats'
        ).fetchall()
        stats_conn.execute('BEGIN')
        try:
          stats_conn.executemany(
            """
            INSERT OR REPLACE INTO chat_user_stats (chat_id, period_type, period_key, sender_ref, sender_name, invoke_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
              (r['chat_id'], r['period_type'], r['period_key'], r['sender_ref'], r['sender_name'], int(r['invoke_count']))
              for r in rows
            ],
          )
          stats_conn.commit()
        except Exception:
          stats_conn.rollback()
          raise
        logger.info('Migrated legacy chat_user_stats rows=%s', len(rows))

      if not _table_has_rows(moderation_conn, 'chat_mutes') and _table_exists(legacy_conn, 'chat_mutes'):
        rows = legacy_conn.execute(
          'SELECT chat_id, sender_ref, muted_at, duration_m FROM chat_mutes'
        ).fetchall()
        moderation_conn.execute('BEGIN')
        try:
          moderation_conn.executemany(
            """
            INSERT OR REPLACE INTO chat_mutes (chat_id, sender_ref, muted_at, duration_m)
            VALUES (?, ?, ?, ?)
            """,
            [(r['chat_id'], r['sender_ref'], r['muted_at'], int(r['duration_m'])) for r in rows],
          )
          moderation_conn.commit()
        except Exception:
          moderation_conn.rollback()
          raise
        logger.info('Migrated legacy chat_mutes rows=%s', len(rows))

      _MIGRATION_DONE = True
    except Exception as e:
      logger.warning('legacy DB migration failed: %s', e)
    finally:
      legacy_conn.close()


def _ensure_split_ready() -> None:
  settings_conn = _get_settings_conn()
  stats_conn = _get_stats_conn()
  moderation_conn = _get_moderation_conn()
  _migrate_from_legacy_if_needed(settings_conn, stats_conn, moderation_conn)


def _get_settings_conn() -> sqlite3.Connection:
  conn: sqlite3.Connection | None = getattr(_LOCAL, 'settings_conn', None)
  if conn is not None:
    return conn
  conn = _new_conn(_resolve_settings_db_path())
  _ensure_settings_tables(conn)
  _LOCAL.settings_conn = conn
  return conn


def _get_stats_conn() -> sqlite3.Connection:
  conn: sqlite3.Connection | None = getattr(_LOCAL, 'stats_conn', None)
  if conn is not None:
    return conn
  conn = _new_conn(_resolve_stats_db_path())
  _ensure_stats_tables(conn)
  _LOCAL.stats_conn = conn
  return conn


def _get_moderation_conn() -> sqlite3.Connection:
  conn: sqlite3.Connection | None = getattr(_LOCAL, 'moderation_conn', None)
  if conn is not None:
    return conn
  conn = _new_conn(_resolve_moderation_db_path())
  _ensure_moderation_tables(conn)
  _LOCAL.moderation_conn = conn
  return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_prompt(chat_id: str) -> Optional[str]:
  """Return the custom prompt for *chat_id*, or ``None`` if not set."""
  with _cache_lock:
    cached = _prompt_cache.get(chat_id, _MISSING)
  if cached is not _MISSING:
    return cached  # type: ignore[return-value]

  _ensure_split_ready()
  conn = _get_settings_conn()
  row = conn.execute(
    'SELECT prompt FROM chat_settings WHERE chat_id = ?', (chat_id,)
  ).fetchone()
  value = row['prompt'] if row is not None else None
  with _cache_lock:
    _prompt_cache[chat_id] = value
  return value


def set_prompt(chat_id: str, prompt: Optional[str]) -> None:
  _ensure_split_ready()
  conn = _get_settings_conn()
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
  logger.info('DB set_prompt chat_id=%s len=%s', chat_id, len(prompt) if prompt else 0)


def get_permission(chat_id: str) -> int:
  """Return the permission level (0-3) for *chat_id*. Default ``0``."""
  with _cache_lock:
    cached = _permission_cache.get(chat_id, _MISSING)
  if cached is not _MISSING:
    return cached  # type: ignore[return-value]

  _ensure_split_ready()
  conn = _get_settings_conn()
  row = conn.execute(
    'SELECT permission FROM chat_settings WHERE chat_id = ?', (chat_id,)
  ).fetchone()
  value = int(row['permission']) if row is not None else 0
  with _cache_lock:
    _permission_cache[chat_id] = value
  return value


def set_permission(chat_id: str, level: int) -> None:
  clamped = max(0, min(3, int(level)))
  _ensure_split_ready()
  conn = _get_settings_conn()
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
  logger.info('DB set_permission chat_id=%s level=%s', chat_id, clamped)


def clear_settings(chat_id: str) -> None:
  """Remove all stored settings for *chat_id*."""
  _ensure_split_ready()
  conn = _get_settings_conn()
  conn.execute('DELETE FROM chat_settings WHERE chat_id = ?', (chat_id,))
  conn.commit()
  with _cache_lock:
    _prompt_cache.pop(chat_id, None)
    _permission_cache.pop(chat_id, None)
    _mode_cache.pop(chat_id, None)
    _triggers_cache.pop(chat_id, None)
    _llm2_model_cache.pop(chat_id, None)


def permission_description(level: int) -> str:
  """Human-readable description of a permission level."""
  mapping = {
    0: 'all moderation FORBIDDEN',
    1: 'delete ALLOWED',
    2: 'delete & mute ALLOWED',
    3: 'delete, mute & kick ALLOWED',
  }
  return mapping.get(level, mapping[0])


def permission_allows_delete(level: int) -> bool:
  return level >= 1


def permission_allows_mute(level: int) -> bool:
  return level >= 2


def permission_allows_kick(level: int) -> bool:
  return level >= 3


# ---------------------------------------------------------------------------
# LLM2 Model Management
# ---------------------------------------------------------------------------

_llm2_model_cache: dict[str, Optional[str]] = {}
_default_llm2_model_cache: Optional[dict] = None


def get_default_llm2_model() -> Optional[dict]:
  """Return the default model (lowest sort_order, is_active=1)."""
  global _default_llm2_model_cache
  if _default_llm2_model_cache is not None:
    return _default_llm2_model_cache
  _ensure_split_ready()
  conn = _get_settings_conn()
  row = conn.execute(
    'SELECT model_id, display_name, description FROM llm_models WHERE is_active = 1 ORDER BY sort_order ASC LIMIT 1'
  ).fetchone()
  if row:
    _default_llm2_model_cache = {
      'model_id': row['model_id'],
      'display_name': row['display_name'],
      'description': row['description'],
    }
  return _default_llm2_model_cache


def get_llm2_model(chat_id: str) -> Optional[str]:
  """Return the model_id for chat_id, or None if not set."""
  with _cache_lock:
    cached = _llm2_model_cache.get(chat_id, _MISSING)
  if cached is not _MISSING:
    return cached if cached is not None else None

  _ensure_split_ready()
  conn = _get_settings_conn()
  row = conn.execute(
    'SELECT llm2_model FROM chat_settings WHERE chat_id = ?', (chat_id,)
  ).fetchone()
  value = row['llm2_model'] if row is not None else None
  with _cache_lock:
    _llm2_model_cache[chat_id] = value
  return value


def set_llm2_model(chat_id: str, model_id: Optional[str]) -> None:
  _ensure_split_ready()
  conn = _get_settings_conn()
  conn.execute(
    """
    INSERT INTO chat_settings (chat_id, llm2_model, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      llm2_model = excluded.llm2_model,
      updated_at = excluded.updated_at
    """,
    (chat_id, model_id),
  )
  conn.commit()
  with _cache_lock:
    _llm2_model_cache[chat_id] = model_id
  logger.info('DB set_llm2_model chat_id=%s model_id=%s', chat_id, model_id)


def get_all_active_models() -> list[dict]:
  """Return all active models ordered by sort_order."""
  _ensure_split_ready()
  conn = _get_settings_conn()
  rows = conn.execute(
    'SELECT model_id, display_name, description, sort_order FROM llm_models WHERE is_active = 1 ORDER BY sort_order ASC'
  ).fetchall()
  return [
    {
      'model_id': row['model_id'],
      'display_name': row['display_name'],
      'description': row['description'],
      'sort_order': row['sort_order'],
    }
    for row in rows
  ]


def get_all_models() -> list[dict]:
  """Return all models (active and inactive) ordered by sort_order."""
  _ensure_split_ready()
  conn = _get_settings_conn()
  rows = conn.execute(
    'SELECT model_id, display_name, description, is_active, sort_order FROM llm_models ORDER BY sort_order ASC'
  ).fetchall()
  return [
    {
      'model_id': row['model_id'],
      'display_name': row['display_name'],
      'description': row['description'],
      'is_active': bool(row['is_active']),
      'sort_order': row['sort_order'],
    }
    for row in rows
  ]


def clear_llm2_model_cache(chat_id: Optional[str] = None) -> None:
  """Clear the LLM2 model cache. If chat_id is provided, only that chat is invalidated. Otherwise, all chats are cleared."""
  global _llm2_model_cache
  with _cache_lock:
    if chat_id is not None:
      if chat_id in _llm2_model_cache:
        del _llm2_model_cache[chat_id]
        logger.debug('Cleared LLM2 model cache for chat_id=%s', chat_id)
    else:
      _llm2_model_cache.clear()
      logger.debug('Cleared all LLM2 model caches')


def clear_default_llm2_model_cache() -> None:
  """Clear the default LLM2 model cache."""
  global _default_llm2_model_cache
  _default_llm2_model_cache = None
  logger.debug('Cleared default LLM2 model cache')


def add_model(model_id: str, display_name: str, description: str = '', sort_order: Optional[int] = None) -> bool:
  """Add a new model. Returns False if model_id already exists."""
  global _default_llm2_model_cache
  _default_llm2_model_cache = None
  _ensure_split_ready()
  conn = _get_settings_conn()
  if sort_order is None:
    max_order_row = conn.execute('SELECT MAX(sort_order) as max_order FROM llm_models').fetchone()
    sort_order = (max_order_row['max_order'] or -1) + 1
  try:
    conn.execute(
      """
      INSERT INTO llm_models (model_id, display_name, description, sort_order)
      VALUES (?, ?, ?, ?)
      """,
      (model_id, display_name, description, sort_order),
    )
    conn.commit()
    logger.info('DB add_model model_id=%s display_name=%s', model_id, display_name)
    return True
  except sqlite3.IntegrityError:
    return False


def update_model(model_id: str, display_name: Optional[str] = None, description: Optional[str] = None, is_active: Optional[bool] = None, sort_order: Optional[int] = None) -> bool:
  """Update a model. Returns False if model_id not found."""
  global _default_llm2_model_cache
  _default_llm2_model_cache = None
  _ensure_split_ready()
  conn = _get_settings_conn()
  existing = conn.execute('SELECT * FROM llm_models WHERE model_id = ?', (model_id,)).fetchone()
  if not existing:
    return False
  updates = []
  values = []
  if display_name is not None:
    updates.append('display_name = ?')
    values.append(display_name)
  if description is not None:
    updates.append('description = ?')
    values.append(description)
  if is_active is not None:
    updates.append('is_active = ?')
    values.append(1 if is_active else 0)
  if sort_order is not None:
    updates.append('sort_order = ?')
    values.append(sort_order)
  if not updates:
    return True
  values.append(model_id)
  conn.execute(f"UPDATE llm_models SET {', '.join(updates)} WHERE model_id = ?", values)
  conn.commit()
  logger.info('DB update_model model_id=%s', model_id)
  return True


def delete_model(model_id: str) -> bool:
  """Delete a model. Returns False if model_id not found."""
  global _default_llm2_model_cache
  _default_llm2_model_cache = None
  _ensure_split_ready()
  conn = _get_settings_conn()
  existing = conn.execute('SELECT model_id FROM llm_models WHERE model_id = ?', (model_id,)).fetchone()
  if not existing:
    return False
  conn.execute('DELETE FROM llm_models WHERE model_id = ?', (model_id,))
  conn.commit()
  logger.info('DB delete_model model_id=%s', model_id)
  return True


# ---------------------------------------------------------------------------
# Mode / Triggers
# ---------------------------------------------------------------------------

def get_mode(chat_id: str) -> str:
  """Return the chat mode ('auto', 'prefix', or 'hybrid'). Default 'prefix'."""
  with _cache_lock:
    cached = _mode_cache.get(chat_id, _MISSING)
  if cached is not _MISSING:
    return cached  # type: ignore[return-value]

  _ensure_split_ready()
  conn = _get_settings_conn()
  row = conn.execute(
    'SELECT mode FROM chat_settings WHERE chat_id = ?', (chat_id,)
  ).fetchone()
  value = row['mode'] if row is not None else DEFAULT_MODE
  if value not in VALID_MODES:
    value = DEFAULT_MODE
  with _cache_lock:
    _mode_cache[chat_id] = value
  return value


def set_mode(chat_id: str, mode: str) -> None:
  if mode not in VALID_MODES:
    mode = DEFAULT_MODE
  _ensure_split_ready()
  conn = _get_settings_conn()
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
  logger.info('DB set_mode chat_id=%s mode=%s', chat_id, mode)


def get_triggers(chat_id: str) -> set[str]:
  """Return the set of enabled trigger types for *chat_id*."""
  with _cache_lock:
    cached = _triggers_cache.get(chat_id, _MISSING)
  if cached is not _MISSING:
    raw = cached  # type: ignore[assignment]
  else:
    _ensure_split_ready()
    conn = _get_settings_conn()
    row = conn.execute(
      'SELECT triggers FROM chat_settings WHERE chat_id = ?', (chat_id,)
    ).fetchone()
    raw = row['triggers'] if row is not None else DEFAULT_TRIGGERS
    with _cache_lock:
      _triggers_cache[chat_id] = raw
  return {t.strip().lower() for t in raw.split(',') if t.strip().lower() in VALID_TRIGGERS}


def set_triggers(chat_id: str, triggers: set[str]) -> None:
  valid = {t for t in triggers if t in VALID_TRIGGERS}
  raw = ','.join(sorted(valid)) if valid else ''
  _ensure_split_ready()
  conn = _get_settings_conn()
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
  logger.info('DB set_triggers chat_id=%s triggers=%s', chat_id, raw)


# ---------------------------------------------------------------------------
# Dashboard stats persistence
# ---------------------------------------------------------------------------

def upsert_stats_batch(rows: list[tuple[str, str, str, str, int]]) -> None:
  """Batch upsert stat counters: [(chat_id, period_type, period_key, stat_key, increment), ...]."""
  if not rows:
    return
  _ensure_split_ready()
  conn = _get_stats_conn()
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
  _ensure_split_ready()
  conn = _get_stats_conn()
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
  _ensure_split_ready()
  conn = _get_stats_conn()
  rows = conn.execute(
    'SELECT stat_key, stat_value FROM chat_stats WHERE chat_id = ? AND period_type = ? AND period_key = ?',
    (chat_id, period_type, period_key),
  ).fetchall()
  return {row['stat_key']: row['stat_value'] for row in rows}


def get_top_users(chat_id: str, period_type: str, period_key: str, limit: int = 5) -> list[tuple[str, str, int]]:
  """Return top users [(sender_ref, sender_name, invoke_count), ...] for a period."""
  _ensure_split_ready()
  conn = _get_stats_conn()
  rows = conn.execute(
    """
    SELECT sender_ref, sender_name, invoke_count FROM chat_user_stats
    WHERE chat_id = ? AND period_type = ? AND period_key = ?
    ORDER BY invoke_count DESC LIMIT ?
    """,
    (chat_id, period_type, period_key, limit),
  ).fetchall()
  return [(row['sender_ref'], row['sender_name'], row['invoke_count']) for row in rows]


# ---------------------------------------------------------------------------
# Mute management
# ---------------------------------------------------------------------------

def _parse_muted_at(muted_at_str: str) -> float:
  """Parse ``datetime('now')`` format to epoch seconds."""
  from datetime import datetime, timezone
  try:
    dt = datetime.strptime(muted_at_str, '%Y-%m-%d %H:%M:%S')
    return dt.replace(tzinfo=timezone.utc).timestamp()
  except (ValueError, TypeError):
    return 0.0


def _is_mute_active(entry: dict) -> bool:
  """Check whether a mute entry is still active."""
  import time
  muted_at_epoch = _parse_muted_at(entry['muted_at'])
  if muted_at_epoch <= 0:
    return False
  expires_at = muted_at_epoch + entry['duration_m'] * 60
  return time.time() < expires_at


def _mute_remaining_minutes(entry: dict) -> int:
  """Return remaining mute minutes (0 if expired)."""
  import time
  muted_at_epoch = _parse_muted_at(entry['muted_at'])
  if muted_at_epoch <= 0:
    return 0
  expires_at = muted_at_epoch + entry['duration_m'] * 60
  remaining = (expires_at - time.time()) / 60
  return max(0, int(remaining))


def add_mute(chat_id: str, sender_ref: str, duration_minutes: int) -> None:
  """Add or update a mute. Persists to DB and updates cache."""
  duration_minutes = max(1, min(1440, int(duration_minutes)))
  _ensure_split_ready()
  conn = _get_moderation_conn()
  conn.execute(
    """
    INSERT INTO chat_mutes (chat_id, sender_ref, muted_at, duration_m)
    VALUES (?, ?, datetime('now'), ?)
    ON CONFLICT(chat_id, sender_ref) DO UPDATE SET
      muted_at = datetime('now'),
      duration_m = excluded.duration_m
    """,
    (chat_id, sender_ref, duration_minutes),
  )
  conn.commit()
  from datetime import datetime, timezone
  now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
  with _cache_lock:
    if chat_id not in _mute_cache:
      _mute_cache[chat_id] = {}
    _mute_cache[chat_id][sender_ref] = {
      'muted_at': now_str,
      'duration_m': duration_minutes,
      'notified': False,
    }
  logger.info('mute added chat_id=%s sender_ref=%s duration=%sm', chat_id, sender_ref, duration_minutes)


def remove_mute(chat_id: str, sender_ref: str) -> None:
  """Remove a mute from DB and cache."""
  _ensure_split_ready()
  conn = _get_moderation_conn()
  conn.execute(
    'DELETE FROM chat_mutes WHERE chat_id = ? AND sender_ref = ?',
    (chat_id, sender_ref),
  )
  conn.commit()
  with _cache_lock:
    if chat_id in _mute_cache:
      _mute_cache[chat_id].pop(sender_ref, None)
  logger.info('mute removed chat_id=%s sender_ref=%s', chat_id, sender_ref)


def clear_mutes(chat_id: str) -> None:
  """Remove all mutes for a chat (used on bot demotion)."""
  _ensure_split_ready()
  conn = _get_moderation_conn()
  conn.execute('DELETE FROM chat_mutes WHERE chat_id = ?', (chat_id,))
  conn.commit()
  with _cache_lock:
    _mute_cache.pop(chat_id, None)
  logger.info('all mutes cleared chat_id=%s', chat_id)


def is_muted(chat_id: str, sender_ref: str) -> bool:
  """Check if a user is currently muted (cache-first, instant)."""
  with _cache_lock:
    chat_mutes = _mute_cache.get(chat_id)
    if chat_mutes is not None:
      entry = chat_mutes.get(sender_ref)
      if entry is not None:
        if _is_mute_active(entry):
          return True
        # Expired — clean up cache
        chat_mutes.pop(sender_ref, None)
        return False
      return False

  _ensure_split_ready()
  conn = _get_moderation_conn()
  row = conn.execute(
    'SELECT muted_at, duration_m FROM chat_mutes WHERE chat_id = ? AND sender_ref = ?',
    (chat_id, sender_ref),
  ).fetchone()
  if row is None:
    return False
  entry = {
    'muted_at': row['muted_at'],
    'duration_m': int(row['duration_m']),
    'notified': False,
  }
  active = _is_mute_active(entry)
  with _cache_lock:
    if chat_id not in _mute_cache:
      _mute_cache[chat_id] = {}
    if active:
      _mute_cache[chat_id][sender_ref] = entry
    else:
      conn.execute(
        'DELETE FROM chat_mutes WHERE chat_id = ? AND sender_ref = ?',
        (chat_id, sender_ref),
      )
      conn.commit()
  return active


def is_mute_notified(chat_id: str, sender_ref: str) -> bool:
  """Check if the first-delete notification was already sent for this mute."""
  with _cache_lock:
    chat_mutes = _mute_cache.get(chat_id, {})
    entry = chat_mutes.get(sender_ref)
    if entry is None:
      return False
    return bool(entry.get('notified'))


def mark_mute_notified(chat_id: str, sender_ref: str) -> None:
  """Mark that the first-delete notification has been sent."""
  with _cache_lock:
    chat_mutes = _mute_cache.get(chat_id, {})
    entry = chat_mutes.get(sender_ref)
    if entry is not None:
      entry['notified'] = True


def get_mute_remaining_minutes(chat_id: str, sender_ref: str) -> int:
  """Return remaining mute minutes for a user (0 if not muted)."""
  with _cache_lock:
    chat_mutes = _mute_cache.get(chat_id, {})
    entry = chat_mutes.get(sender_ref)
    if entry is not None:
      return _mute_remaining_minutes(entry)
  return 0
