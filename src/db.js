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
let _db = null;
let _dbPath = null;
let _statsDirty = false;
let _statsSaveInterval = null;

function getDbPath() {
  if (_dbPath) return _dbPath;
  const dataDir = config.dataDir || path.resolve(process.cwd(), 'data');
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }
  _dbPath = path.join(dataDir, 'bot.db');
  return _dbPath;
}

function saveDb() {
  if (!_db) return;
  try {
    const data = _db.export();
    const buffer = Buffer.from(data);
    fs.writeFileSync(_dbPath, buffer);
  } catch (err) {
    logger.error({ err }, 'DB save failed');
  }
}

function startStatsSaveInterval() {
  if (_statsSaveInterval) return;
  _statsSaveInterval = setInterval(() => {
    if (_statsDirty) {
      saveDb();
      _statsDirty = false;
      logger.debug('DB periodic save (stats)');
    }
  }, 3 * 60 * 1000);
}

async function init() {
  if (_db) return;

  _sql = await initSqlJs();

  const dbPath = getDbPath();
  let data = null;

  if (fs.existsSync(dbPath)) {
    try {
      const fileBuffer = fs.readFileSync(dbPath);
      data = new Uint8Array(fileBuffer);
    } catch (err) {
      logger.warn({ err, dbPath }, 'Could not read existing DB, creating new');
    }
  }

  _db = new _sql.Database(data);
  initTables();
  startStatsSaveInterval();

  logger.info({ dbPath }, 'DB initialized');
}

function initTables() {
  _db.run(`
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

  _db.run(`
    CREATE TABLE IF NOT EXISTS chat_stats (
      chat_id      TEXT NOT NULL,
      period_type  TEXT NOT NULL,
      period_key   TEXT NOT NULL,
      stat_key     TEXT NOT NULL,
      stat_value   INTEGER NOT NULL DEFAULT 0,
      PRIMARY KEY (chat_id, period_type, period_key, stat_key)
    )
  `);

  _db.run(`
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

  _db.run(`
    CREATE TABLE IF NOT EXISTS llm_models (
      model_id     TEXT PRIMARY KEY,
      display_name TEXT NOT NULL,
      description  TEXT,
      is_active    INTEGER NOT NULL DEFAULT 1,
      sort_order   INTEGER NOT NULL DEFAULT 0
    )
  `);
}

function runQuery(sql, ...params) {
  _db.run(sql, params);
}

function getOne(sql, ...params) {
  const stmt = _db.prepare(sql);
  stmt.bind(params);
  if (stmt.step()) {
    const row = stmt.getAsObject();
    stmt.free();
    return row;
  }
  stmt.free();
  return null;
}

function getAll(sql, ...params) {
  const stmt = _db.prepare(sql);
  stmt.bind(params);
  const results = [];
  while (stmt.step()) {
    results.push(stmt.getAsObject());
  }
  stmt.free();
  return results;
}

function getPrompt(chatId) {
  const row = getOne('SELECT prompt FROM chat_settings WHERE chat_id = ?', chatId);
  return row?.prompt ?? null;
}

function setPrompt(chatId, prompt) {
  runQuery(`
    INSERT INTO chat_settings (chat_id, prompt, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      prompt = excluded.prompt,
      updated_at = excluded.updated_at
  `, chatId, prompt);
  saveDb();
  logger.info({ chatId, promptLen: prompt?.length || 0 }, 'DB set_prompt');
}

function getPermission(chatId) {
  const row = getOne('SELECT permission FROM chat_settings WHERE chat_id = ?', chatId);
  return row?.permission ?? 0;
}

function setPermission(chatId, level) {
  const clamped = Math.max(0, Math.min(3, parseInt(level, 10) || 0));
  runQuery(`
    INSERT INTO chat_settings (chat_id, permission, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      permission = excluded.permission,
      updated_at = excluded.updated_at
  `, chatId, clamped);
  saveDb();
  logger.info({ chatId, level: clamped }, 'DB set_permission');
}

function getMode(chatId) {
  const row = getOne('SELECT mode FROM chat_settings WHERE chat_id = ?', chatId);
  let value = row?.mode ?? DEFAULT_MODE;
  if (!VALID_MODES.has(value)) value = DEFAULT_MODE;
  return value;
}

function setMode(chatId, mode) {
  if (!VALID_MODES.has(mode)) mode = DEFAULT_MODE;
  runQuery(`
    INSERT INTO chat_settings (chat_id, mode, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      mode = excluded.mode,
      updated_at = excluded.updated_at
  `, chatId, mode);
  saveDb();
  logger.info({ chatId, mode }, 'DB set_mode');
}

function getTriggers(chatId) {
  const row = getOne('SELECT triggers FROM chat_settings WHERE chat_id = ?', chatId);
  const raw = row?.triggers ?? DEFAULT_TRIGGERS;
  return new Set(raw.split(',').filter((t) => VALID_TRIGGERS.has(t.trim().toLowerCase())).map((t) => t.trim().toLowerCase()));
}

function setTriggers(chatId, triggers) {
  const valid = [...triggers].filter((t) => VALID_TRIGGERS.has(t));
  const raw = valid.sort().join(',') || '';
  runQuery(`
    INSERT INTO chat_settings (chat_id, triggers, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      triggers = excluded.triggers,
      updated_at = excluded.updated_at
  `, chatId, raw);
  saveDb();
  logger.info({ chatId, triggers: raw }, 'DB set_triggers');
}

function clearSettings(chatId) {
  runQuery('DELETE FROM chat_settings WHERE chat_id = ?', chatId);
  saveDb();
  logger.info({ chatId }, 'DB clear_settings');
}

function getStats(chatId, periodType, periodKey) {
  const rows = getAll(
    'SELECT stat_key, stat_value FROM chat_stats WHERE chat_id = ? AND period_type = ? AND period_key = ?',
    chatId, periodType, periodKey
  );
  const result = {};
  for (const row of rows) {
    result[row.stat_key] = row.stat_value;
  }
  return result;
}

function getTopUsers(chatId, periodType, periodKey, limit = 5) {
  const rows = getAll(
    `SELECT sender_ref, sender_name, invoke_count FROM chat_user_stats
     WHERE chat_id = ? AND period_type = ? AND period_key = ?
     ORDER BY invoke_count DESC LIMIT ?`,
    chatId, periodType, periodKey, limit
  );
  return rows.map((row) => ({ senderRef: row.sender_ref, senderName: row.sender_name, invokeCount: row.invoke_count }));
}

function getDefaultLlm2Model() {
  const row = getOne(
    'SELECT model_id, display_name, description FROM llm_models WHERE is_active = 1 ORDER BY sort_order ASC LIMIT 1'
  );
  if (row) {
    return { modelId: row.model_id, displayName: row.display_name, description: row.description };
  }
  return null;
}

function getLlm2Model(chatId) {
  const row = getOne('SELECT llm2_model FROM chat_settings WHERE chat_id = ?', chatId);
  return row?.llm2_model ?? null;
}

function setLlm2Model(chatId, modelId) {
  runQuery(`
    INSERT INTO chat_settings (chat_id, llm2_model, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      llm2_model = excluded.llm2_model,
      updated_at = excluded.updated_at
  `, chatId, modelId);
  saveDb();
  logger.info({ chatId, modelId }, 'DB set_llm2_model');
}

function getAllActiveModels() {
  const rows = getAll(
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
  const rows = getAll(
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
    const maxOrder = getOne('SELECT MAX(sort_order) as max_order FROM llm_models');
    sortOrder = (maxOrder?.max_order ?? -1) + 1;
  }
  try {
    runQuery(
      'INSERT INTO llm_models (model_id, display_name, description, sort_order) VALUES (?, ?, ?, ?)',
      modelId, displayName, description, sortOrder
    );
    saveDb();
    logger.info({ modelId, displayName }, 'DB add_model');
    return true;
  } catch (err) {
    if (err.message?.includes('UNIQUE constraint failed') || err.code === 'SQLITE_CONSTRAINT_PRIMARYKEY') {
      return false;
    }
    throw err;
  }
}

function updateModel(modelId, { displayName, description, isActive, sortOrder } = {}) {
  const existing = getOne('SELECT model_id FROM llm_models WHERE model_id = ?', modelId);
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
  runQuery(`UPDATE llm_models SET ${updates.join(', ')} WHERE model_id = ?`, ...values);
  saveDb();
  logger.info({ modelId }, 'DB update_model');
  return true;
}

function deleteModel(modelId) {
  const existing = getOne('SELECT model_id FROM llm_models WHERE model_id = ?', modelId);
  if (!existing) return false;
  runQuery('DELETE FROM llm_models WHERE model_id = ?', modelId);
  saveDb();
  logger.info({ modelId }, 'DB delete_model');
  return true;
}

export {
  init,
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
