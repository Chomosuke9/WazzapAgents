import initSqlJs from 'sql.js';
import path from 'path';
import fs from 'fs';
import logger from './logger.js';
import config from './config.js';

const VALID_MODES = new Set(['auto', 'prefix', 'hybrid']);
const DEFAULT_MODE = 'prefix';
const VALID_TRIGGERS = new Set(['tag', 'reply', 'join', 'name']);
const DEFAULT_TRIGGERS = 'tag,reply,name';

let _sql = null;

const _settingsState = { db: null, dbPath: null, lastLoadedMtimeMs: 0 };
const _statsState = { db: null, dbPath: null, lastLoadedMtimeMs: 0 };
const _moderationState = { db: null, dbPath: null, lastLoadedMtimeMs: 0 };

function ensureParentDir(filePath) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function getSettingsDbPath() {
  if (_settingsState.dbPath) return _settingsState.dbPath;
  _settingsState.dbPath = config.settingsDbPath;
  ensureParentDir(_settingsState.dbPath);
  return _settingsState.dbPath;
}

function getStatsDbPath() {
  if (_statsState.dbPath) return _statsState.dbPath;
  _statsState.dbPath = config.statsDbPath;
  ensureParentDir(_statsState.dbPath);
  return _statsState.dbPath;
}

function getModerationDbPath() {
  if (_moderationState.dbPath) return _moderationState.dbPath;
  _moderationState.dbPath = config.moderationDbPath;
  ensureParentDir(_moderationState.dbPath);
  return _moderationState.dbPath;
}

function getFileMtimeMs(filePath) {
  try {
    return fs.statSync(filePath).mtimeMs || 0;
  } catch {
    return 0;
  }
}

function loadDbDataFromDisk(dbPath) {
  if (!fs.existsSync(dbPath)) return null;
  const fileBuffer = fs.readFileSync(dbPath);
  return new Uint8Array(fileBuffer);
}

function replaceDb(state, data, initTablesFn) {
  state.db = new _sql.Database(data);
  initTablesFn(state.db);
}

function refreshDbFromDiskIfChanged(state, initTablesFn) {
  if (!_sql || !state.db || !state.dbPath) return;
  const diskMtimeMs = getFileMtimeMs(state.dbPath);
  if (!diskMtimeMs || diskMtimeMs <= state.lastLoadedMtimeMs) return;
  try {
    const data = loadDbDataFromDisk(state.dbPath);
    replaceDb(state, data, initTablesFn);
    state.lastLoadedMtimeMs = diskMtimeMs;
    logger.debug({ dbPath: state.dbPath }, 'DB refreshed from disk');
  } catch (err) {
    logger.warn({ err, dbPath: state.dbPath }, 'DB refresh from disk failed');
  }
}

function saveDb(state) {
  if (!state.db || !state.dbPath) return;
  try {
    const data = state.db.export();
    const buffer = Buffer.from(data);
    const tempPath = `${state.dbPath}.tmp`;
    fs.writeFileSync(tempPath, buffer);
    fs.renameSync(tempPath, state.dbPath);
    state.lastLoadedMtimeMs = getFileMtimeMs(state.dbPath) || Date.now();
  } catch (err) {
    logger.error({ err, dbPath: state.dbPath }, 'DB save failed');
  }
}

function initSettingsTables(db) {
  db.run(`
    CREATE TABLE IF NOT EXISTS chat_settings (
      chat_id    TEXT PRIMARY KEY,
      prompt     TEXT,
      permission INTEGER NOT NULL DEFAULT 0,
      mode       TEXT NOT NULL DEFAULT '${DEFAULT_MODE}',
      triggers   TEXT NOT NULL DEFAULT '${DEFAULT_TRIGGERS}',
      llm2_model TEXT,
      updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
  `);

  db.run(`
    CREATE TABLE IF NOT EXISTS llm_models (
      model_id     TEXT PRIMARY KEY,
      display_name TEXT NOT NULL,
      description  TEXT,
      is_active    INTEGER NOT NULL DEFAULT 1,
      sort_order   INTEGER NOT NULL DEFAULT 0
    )
  `);
}

function initStatsTables(db) {
  db.run(`
    CREATE TABLE IF NOT EXISTS chat_stats (
      chat_id      TEXT NOT NULL,
      period_type  TEXT NOT NULL,
      period_key   TEXT NOT NULL,
      stat_key     TEXT NOT NULL,
      stat_value   INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (chat_id, period_type, period_key, stat_key)
    )
  `);

  db.run(`
    CREATE TABLE IF NOT EXISTS chat_user_stats (
      chat_id      TEXT NOT NULL,
      period_type  TEXT NOT NULL,
      period_key   TEXT NOT NULL,
      sender_ref   TEXT NOT NULL,
      sender_name  TEXT NOT NULL DEFAULT '',
      invoke_count INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (chat_id, period_type, period_key, sender_ref)
    )
  `);
}

function initModerationTables(db) {
  db.run(`
    CREATE TABLE IF NOT EXISTS chat_mutes (
      chat_id     TEXT NOT NULL,
      sender_ref  TEXT NOT NULL,
      muted_at    TEXT NOT NULL DEFAULT (datetime('now')),
      duration_m  INTEGER NOT NULL,
      PRIMARY KEY (chat_id, sender_ref)
    )
  `);
}

function queryRows(db, sql, ...params) {
  const stmt = db.prepare(sql);
  stmt.bind(params);
  const rows = [];
  while (stmt.step()) rows.push(stmt.getAsObject());
  stmt.free();
  return rows;
}

function escapeIdentifier(identifier) {
  return `"${String(identifier).replace(/"/g, '""')}"`;
}

function tableExists(db, tableName) {
  const rows = queryRows(
    db,
    'SELECT 1 AS ok FROM sqlite_master WHERE type = ? AND name = ? LIMIT 1',
    'table',
    tableName
  );
  return rows.length > 0;
}

function hasRows(db, tableName) {
  if (!tableExists(db, tableName)) return false;
  const rows = queryRows(db, `SELECT 1 AS ok FROM ${escapeIdentifier(tableName)} LIMIT 1`);
  return rows.length > 0;
}

function getColumns(db, tableName) {
  if (!tableExists(db, tableName)) return new Set();
  const rows = queryRows(db, `PRAGMA table_info(${escapeIdentifier(tableName)})`);
  return new Set(rows.map((r) => String(r.name)));
}

function migrateFromLegacyIfNeeded() {
  const legacyDbPath = path.join(config.dataDir, 'bot.db');
  if (!fs.existsSync(legacyDbPath)) return;

  const settingsPath = getSettingsDbPath();
  const statsPath = getStatsDbPath();
  const moderationPath = getModerationDbPath();
  const normalizedLegacy = path.resolve(legacyDbPath);
  if ([settingsPath, statsPath, moderationPath].some((p) => path.resolve(p) === normalizedLegacy)) {
    return;
  }

  let legacy = null;
  try {
    legacy = new _sql.Database(loadDbDataFromDisk(legacyDbPath));
  } catch (err) {
    logger.warn({ err, legacyDbPath }, 'Failed opening legacy bot.db for migration');
    return;
  }

  try {
    const legacyChatSettingsColumns = getColumns(legacy, 'chat_settings');

    if (_settingsState.db && !hasRows(_settingsState.db, 'chat_settings') && legacyChatSettingsColumns.size > 0) {
      const chatSettingsRows = queryRows(legacy, `
        SELECT
          chat_id,
          prompt,
          COALESCE(permission, 0) AS permission,
          ${legacyChatSettingsColumns.has('mode') ? 'mode' : `'${DEFAULT_MODE}'`} AS mode,
          ${legacyChatSettingsColumns.has('triggers') ? 'triggers' : `'${DEFAULT_TRIGGERS}'`} AS triggers,
          ${legacyChatSettingsColumns.has('llm2_model') ? 'llm2_model' : 'NULL'} AS llm2_model,
          COALESCE(updated_at, datetime('now')) AS updated_at
        FROM chat_settings
      `);
      _settingsState.db.run('BEGIN TRANSACTION');
      try {
        for (const row of chatSettingsRows) {
          _settingsState.db.run(`
            INSERT INTO chat_settings (chat_id, prompt, permission, mode, triggers, llm2_model, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
              prompt = excluded.prompt,
              permission = excluded.permission,
              mode = excluded.mode,
              triggers = excluded.triggers,
              llm2_model = excluded.llm2_model,
              updated_at = excluded.updated_at
          `, [row.chat_id, row.prompt, row.permission, row.mode, row.triggers, row.llm2_model, row.updated_at]);
        }
        _settingsState.db.run('COMMIT');
      } catch (err) {
        _settingsState.db.run('ROLLBACK');
        throw err;
      }
      if (chatSettingsRows.length > 0) saveDb(_settingsState);
      logger.info({ rows: chatSettingsRows.length }, 'Migrated legacy chat_settings to settings.db');
    }

    if (_settingsState.db && !hasRows(_settingsState.db, 'llm_models') && tableExists(legacy, 'llm_models')) {
      const llmRows = queryRows(legacy, `
        SELECT model_id, display_name, description, COALESCE(is_active, 1) AS is_active, COALESCE(sort_order, 0) AS sort_order
        FROM llm_models
      `);
      _settingsState.db.run('BEGIN TRANSACTION');
      try {
        for (const row of llmRows) {
          _settingsState.db.run(`
            INSERT OR REPLACE INTO llm_models (model_id, display_name, description, is_active, sort_order)
            VALUES (?, ?, ?, ?, ?)
          `, [row.model_id, row.display_name, row.description, row.is_active, row.sort_order]);
        }
        _settingsState.db.run('COMMIT');
      } catch (err) {
        _settingsState.db.run('ROLLBACK');
        throw err;
      }
      if (llmRows.length > 0) saveDb(_settingsState);
      logger.info({ rows: llmRows.length }, 'Migrated legacy llm_models to settings.db');
    }

    if (_statsState.db && !hasRows(_statsState.db, 'chat_stats') && tableExists(legacy, 'chat_stats')) {
      const statRows = queryRows(legacy, `
        SELECT chat_id, period_type, period_key, stat_key, stat_value
        FROM chat_stats
      `);
      _statsState.db.run('BEGIN TRANSACTION');
      try {
        for (const row of statRows) {
          _statsState.db.run(`
            INSERT OR REPLACE INTO chat_stats (chat_id, period_type, period_key, stat_key, stat_value)
            VALUES (?, ?, ?, ?, ?)
          `, [row.chat_id, row.period_type, row.period_key, row.stat_key, row.stat_value]);
        }
        _statsState.db.run('COMMIT');
      } catch (err) {
        _statsState.db.run('ROLLBACK');
        throw err;
      }
      if (statRows.length > 0) saveDb(_statsState);
      logger.info({ rows: statRows.length }, 'Migrated legacy chat_stats to stats.db');
    }

    if (_statsState.db && !hasRows(_statsState.db, 'chat_user_stats') && tableExists(legacy, 'chat_user_stats')) {
      const userRows = queryRows(legacy, `
        SELECT chat_id, period_type, period_key, sender_ref, sender_name, invoke_count
        FROM chat_user_stats
      `);
      _statsState.db.run('BEGIN TRANSACTION');
      try {
        for (const row of userRows) {
          _statsState.db.run(`
            INSERT OR REPLACE INTO chat_user_stats (chat_id, period_type, period_key, sender_ref, sender_name, invoke_count)
            VALUES (?, ?, ?, ?, ?, ?)
          `, [row.chat_id, row.period_type, row.period_key, row.sender_ref, row.sender_name, row.invoke_count]);
        }
        _statsState.db.run('COMMIT');
      } catch (err) {
        _statsState.db.run('ROLLBACK');
        throw err;
      }
      if (userRows.length > 0) saveDb(_statsState);
      logger.info({ rows: userRows.length }, 'Migrated legacy chat_user_stats to stats.db');
    }

    if (_moderationState.db && !hasRows(_moderationState.db, 'chat_mutes') && tableExists(legacy, 'chat_mutes')) {
      const muteRows = queryRows(legacy, `
        SELECT chat_id, sender_ref, muted_at, duration_m
        FROM chat_mutes
      `);
      _moderationState.db.run('BEGIN TRANSACTION');
      try {
        for (const row of muteRows) {
          _moderationState.db.run(`
            INSERT OR REPLACE INTO chat_mutes (chat_id, sender_ref, muted_at, duration_m)
            VALUES (?, ?, ?, ?)
          `, [row.chat_id, row.sender_ref, row.muted_at, row.duration_m]);
        }
        _moderationState.db.run('COMMIT');
      } catch (err) {
        _moderationState.db.run('ROLLBACK');
        throw err;
      }
      if (muteRows.length > 0) saveDb(_moderationState);
      logger.info({ rows: muteRows.length }, 'Migrated legacy chat_mutes to moderation.db');
    }
  } catch (err) {
    logger.warn({ err }, 'Legacy DB migration skipped due to error');
  }
}

async function init() {
  if (_settingsState.db && _statsState.db && _moderationState.db) return;

  _sql = await initSqlJs();

  const settingsPath = getSettingsDbPath();
  const statsPath = getStatsDbPath();
  const moderationPath = getModerationDbPath();

  let settingsData = null;
  let statsData = null;
  let moderationData = null;
  try {
    settingsData = loadDbDataFromDisk(settingsPath);
  } catch (err) {
    logger.warn({ err, dbPath: settingsPath }, 'Could not read settings DB, creating new');
  }
  try {
    statsData = loadDbDataFromDisk(statsPath);
  } catch (err) {
    logger.warn({ err, dbPath: statsPath }, 'Could not read stats DB, creating new');
  }
  try {
    moderationData = loadDbDataFromDisk(moderationPath);
  } catch (err) {
    logger.warn({ err, dbPath: moderationPath }, 'Could not read moderation DB, creating new');
  }

  replaceDb(_settingsState, settingsData, initSettingsTables);
  replaceDb(_statsState, statsData, initStatsTables);
  replaceDb(_moderationState, moderationData, initModerationTables);
  _settingsState.lastLoadedMtimeMs = getFileMtimeMs(settingsPath) || Date.now();
  _statsState.lastLoadedMtimeMs = getFileMtimeMs(statsPath) || Date.now();
  _moderationState.lastLoadedMtimeMs = getFileMtimeMs(moderationPath) || Date.now();

  migrateFromLegacyIfNeeded();

  logger.info({ settingsPath, statsPath, moderationPath }, 'DB initialized (split)');
}

function runSettingsQuery(sql, ...params) {
  refreshDbFromDiskIfChanged(_settingsState, initSettingsTables);
  _settingsState.db.run(sql, params);
}

function runStatsQuery(sql, ...params) {
  refreshDbFromDiskIfChanged(_statsState, initStatsTables);
  _statsState.db.run(sql, params);
}

function getOneFromState(state, initTablesFn, sql, ...params) {
  refreshDbFromDiskIfChanged(state, initTablesFn);
  const stmt = state.db.prepare(sql);
  stmt.bind(params);
  if (stmt.step()) {
    const row = stmt.getAsObject();
    stmt.free();
    return row;
  }
  stmt.free();
  return null;
}

function getAllFromState(state, initTablesFn, sql, ...params) {
  refreshDbFromDiskIfChanged(state, initTablesFn);
  const stmt = state.db.prepare(sql);
  stmt.bind(params);
  const results = [];
  while (stmt.step()) {
    results.push(stmt.getAsObject());
  }
  stmt.free();
  return results;
}

function getPrompt(chatId) {
  const row = getOneFromState(_settingsState, initSettingsTables, 'SELECT prompt FROM chat_settings WHERE chat_id = ?', chatId);
  return row?.prompt ?? null;
}

function setPrompt(chatId, prompt) {
  runSettingsQuery(`
    INSERT INTO chat_settings (chat_id, prompt, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      prompt = excluded.prompt,
      updated_at = excluded.updated_at
  `, chatId, prompt);
  saveDb(_settingsState);
  logger.info({ chatId, promptLen: prompt?.length || 0 }, 'DB set_prompt');
}

function getPermission(chatId) {
  const row = getOneFromState(_settingsState, initSettingsTables, 'SELECT permission FROM chat_settings WHERE chat_id = ?', chatId);
  return row?.permission ?? 0;
}

function setPermission(chatId, level) {
  const clamped = Math.max(0, Math.min(3, parseInt(level, 10) || 0));
  runSettingsQuery(`
    INSERT INTO chat_settings (chat_id, permission, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      permission = excluded.permission,
      updated_at = excluded.updated_at
  `, chatId, clamped);
  saveDb(_settingsState);
  logger.info({ chatId, level: clamped }, 'DB set_permission');
}

function getMode(chatId) {
  const row = getOneFromState(_settingsState, initSettingsTables, 'SELECT mode FROM chat_settings WHERE chat_id = ?', chatId);
  let value = row?.mode ?? DEFAULT_MODE;
  if (!VALID_MODES.has(value)) value = DEFAULT_MODE;
  return value;
}

function setMode(chatId, mode) {
  if (!VALID_MODES.has(mode)) mode = DEFAULT_MODE;
  runSettingsQuery(`
    INSERT INTO chat_settings (chat_id, mode, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      mode = excluded.mode,
      updated_at = excluded.updated_at
  `, chatId, mode);
  saveDb(_settingsState);
  logger.info({ chatId, mode }, 'DB set_mode');
}

function getTriggers(chatId) {
  const row = getOneFromState(_settingsState, initSettingsTables, 'SELECT triggers FROM chat_settings WHERE chat_id = ?', chatId);
  const raw = row?.triggers ?? DEFAULT_TRIGGERS;
  return new Set(raw.split(',').filter((t) => VALID_TRIGGERS.has(t.trim().toLowerCase())).map((t) => t.trim().toLowerCase()));
}

function setTriggers(chatId, triggers) {
  const valid = [...triggers].filter((t) => VALID_TRIGGERS.has(t));
  const raw = valid.sort().join(',') || '';
  runSettingsQuery(`
    INSERT INTO chat_settings (chat_id, triggers, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      triggers = excluded.triggers,
      updated_at = excluded.updated_at
  `, chatId, raw);
  saveDb(_settingsState);
  logger.info({ chatId, triggers: raw }, 'DB set_triggers');
}

function clearSettings(chatId) {
  runSettingsQuery('DELETE FROM chat_settings WHERE chat_id = ?', chatId);
  saveDb(_settingsState);
  logger.info({ chatId }, 'DB clear_settings');
}

function getStats(chatId, periodType, periodKey) {
  const rows = getAllFromState(
    _statsState,
    initStatsTables,
    'SELECT stat_key, stat_value FROM chat_stats WHERE chat_id = ? AND period_type = ? AND period_key = ?',
    chatId,
    periodType,
    periodKey
  );
  const result = {};
  for (const row of rows) result[row.stat_key] = row.stat_value;
  return result;
}

function getTopUsers(chatId, periodType, periodKey, limit = 5) {
  const rows = getAllFromState(
    _statsState,
    initStatsTables,
    `SELECT sender_ref, sender_name, invoke_count FROM chat_user_stats
     WHERE chat_id = ? AND period_type = ? AND period_key = ?
     ORDER BY invoke_count DESC LIMIT ?`,
    chatId,
    periodType,
    periodKey,
    limit
  );
  return rows.map((row) => ({ senderRef: row.sender_ref, senderName: row.sender_name, invokeCount: row.invoke_count }));
}

function getDefaultLlm2Model() {
  const row = getOneFromState(
    _settingsState,
    initSettingsTables,
    'SELECT model_id, display_name, description FROM llm_models WHERE is_active = 1 ORDER BY sort_order ASC LIMIT 1'
  );
  if (row) return { modelId: row.model_id, displayName: row.display_name, description: row.description };
  return null;
}

function getLlm2Model(chatId) {
  const row = getOneFromState(_settingsState, initSettingsTables, 'SELECT llm2_model FROM chat_settings WHERE chat_id = ?', chatId);
  return row?.llm2_model ?? null;
}

function setLlm2Model(chatId, modelId) {
  runSettingsQuery(`
    INSERT INTO chat_settings (chat_id, llm2_model, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      llm2_model = excluded.llm2_model,
      updated_at = excluded.updated_at
  `, chatId, modelId);
  saveDb(_settingsState);
  logger.info({ chatId, modelId }, 'DB set_llm2_model');
}

function getAllActiveModels() {
  const rows = getAllFromState(
    _settingsState,
    initSettingsTables,
    'SELECT model_id, display_name, description, sort_order FROM llm_models WHERE is_active = 1 ORDER BY sort_order ASC'
  );
  return rows.map((row) => ({
    modelId: row.model_id,
    displayName: row.display_name,
    description: row.description,
    sortOrder: row.sort_order,
  }));
}

function getAllModels() {
  const rows = getAllFromState(
    _settingsState,
    initSettingsTables,
    'SELECT model_id, display_name, description, is_active, sort_order FROM llm_models ORDER BY sort_order ASC'
  );
  return rows.map((row) => ({
    modelId: row.model_id,
    displayName: row.display_name,
    description: row.description,
    isActive: Boolean(row.is_active),
    sortOrder: row.sort_order,
  }));
}

function addModel(modelId, displayName, description = '', sortOrder = null) {
  if (sortOrder === null) {
    const maxOrder = getOneFromState(_settingsState, initSettingsTables, 'SELECT MAX(sort_order) as max_order FROM llm_models');
    sortOrder = (maxOrder?.max_order ?? -1) + 1;
  }
  try {
    runSettingsQuery(
      'INSERT INTO llm_models (model_id, display_name, description, sort_order) VALUES (?, ?, ?, ?)',
      modelId,
      displayName,
      description,
      sortOrder
    );
    saveDb(_settingsState);
    logger.info({ modelId, displayName }, 'DB add_model');
    return true;
  } catch (err) {
    if (err.message?.includes('UNIQUE constraint failed') || err.code === 'SQLITE_CONSTRAINT_PRIMARYKEY') return false;
    throw err;
  }
}

function updateModel(modelId, { displayName, description, isActive, sortOrder } = {}) {
  const existing = getOneFromState(_settingsState, initSettingsTables, 'SELECT model_id FROM llm_models WHERE model_id = ?', modelId);
  if (!existing) return false;
  const updates = [];
  const values = [];
  if (displayName !== undefined) {
    updates.push('display_name = ?');
    values.push(displayName);
  }
  if (description !== undefined) {
    updates.push('description = ?');
    values.push(description);
  }
  if (isActive !== undefined) {
    updates.push('is_active = ?');
    values.push(isActive ? 1 : 0);
  }
  if (sortOrder !== undefined) {
    updates.push('sort_order = ?');
    values.push(sortOrder);
  }
  if (updates.length === 0) return true;
  values.push(modelId);
  runSettingsQuery(`UPDATE llm_models SET ${updates.join(', ')} WHERE model_id = ?`, ...values);
  saveDb(_settingsState);
  logger.info({ modelId }, 'DB update_model');
  return true;
}

function deleteModel(modelId) {
  const existing = getOneFromState(_settingsState, initSettingsTables, 'SELECT model_id FROM llm_models WHERE model_id = ?', modelId);
  if (!existing) return false;
  runSettingsQuery('DELETE FROM llm_models WHERE model_id = ?', modelId);
  saveDb(_settingsState);
  logger.info({ modelId }, 'DB delete_model');
  return true;
}

function getDbPath() {
  return getSettingsDbPath();
}

export {
  init,
  getDbPath,
  getSettingsDbPath,
  getStatsDbPath,
  getModerationDbPath,
  getPrompt,
  setPrompt,
  getPermission,
  setPermission,
  getMode,
  setMode,
  getTriggers,
  setTriggers,
  clearSettings,
  getStats,
  getTopUsers,
  getLlm2Model,
  setLlm2Model,
  getAllActiveModels,
  getAllModels,
  getDefaultLlm2Model,
  addModel,
  updateModel,
  deleteModel,
  VALID_MODES,
  DEFAULT_MODE,
  VALID_TRIGGERS,
  DEFAULT_TRIGGERS,
};
