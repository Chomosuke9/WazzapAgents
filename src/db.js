import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import logger from './logger.js';
import config from './config.js';

let db = null;
const cache = {
  prompt: new Map(),
  permission: new Map(),
  mode: new Map(),
  triggers: new Map(),
};

const VALID_MODES = new Set(['auto', 'prefix', 'hybrid']);
const DEFAULT_MODE = 'prefix';
const VALID_TRIGGERS = new Set(['tag', 'reply', 'join', 'name']);
const DEFAULT_TRIGGERS = 'tag,reply,name';

function getDbPath() {
  const dataDir = config.dataDir || path.resolve(process.cwd(), 'data');
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }
  return path.join(dataDir, 'bot.db');
}

function getDb() {
  if (db) return db;
  const dbPath = getDbPath();
  db = new Database(dbPath);
  db.pragma('journal_mode = WAL');
  db.pragma('busy_timeout = 3000');
  initTables();
  return db;
}

function initTables() {
  const db = getDb();
  db.exec(`
    CREATE TABLE IF NOT EXISTS chat_settings (
      chat_id    TEXT PRIMARY KEY,
      prompt     TEXT,
      permission INTEGER NOT NULL DEFAULT 0,
      mode       TEXT NOT NULL DEFAULT '${DEFAULT_MODE}',
      triggers   TEXT NOT NULL DEFAULT '${DEFAULT_TRIGGERS}',
      llm2_model TEXT,
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

    CREATE TABLE IF NOT EXISTS llm_models (
      model_id     TEXT PRIMARY KEY,
      display_name TEXT NOT NULL,
      description  TEXT,
      is_active    INTEGER NOT NULL DEFAULT 1,
      sort_order   INTEGER NOT NULL DEFAULT 0
    );
  `);
}

function getPrompt(chatId) {
  if (cache.prompt.has(chatId)) {
    return cache.prompt.get(chatId);
  }
  const db = getDb();
  const row = db.prepare('SELECT prompt FROM chat_settings WHERE chat_id = ?').get(chatId);
  const value = row?.prompt ?? null;
  cache.prompt.set(chatId, value);
  return value;
}

function setPrompt(chatId, prompt) {
  const db = getDb();
  db.prepare(`
    INSERT INTO chat_settings (chat_id, prompt, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      prompt = excluded.prompt,
      updated_at = excluded.updated_at
  `).run(chatId, prompt);
  cache.prompt.set(chatId, prompt);
  logger.info({ chatId, promptLen: prompt?.length || 0 }, 'DB set_prompt');
}

function getPermission(chatId) {
  if (cache.permission.has(chatId)) {
    return cache.permission.get(chatId);
  }
  const db = getDb();
  const row = db.prepare('SELECT permission FROM chat_settings WHERE chat_id = ?').get(chatId);
  const value = row?.permission ?? 0;
  cache.permission.set(chatId, value);
  return value;
}

function setPermission(chatId, level) {
  const clamped = Math.max(0, Math.min(3, parseInt(level, 10) || 0));
  const db = getDb();
  db.prepare(`
    INSERT INTO chat_settings (chat_id, permission, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      permission = excluded.permission,
      updated_at = excluded.updated_at
  `).run(chatId, clamped);
  cache.permission.set(chatId, clamped);
  logger.info({ chatId, level: clamped }, 'DB set_permission');
}

function getMode(chatId) {
  if (cache.mode.has(chatId)) {
    return cache.mode.get(chatId);
  }
  const db = getDb();
  const row = db.prepare('SELECT mode FROM chat_settings WHERE chat_id = ?').get(chatId);
  let value = row?.mode ?? DEFAULT_MODE;
  if (!VALID_MODES.has(value)) value = DEFAULT_MODE;
  cache.mode.set(chatId, value);
  return value;
}

function setMode(chatId, mode) {
  if (!VALID_MODES.has(mode)) mode = DEFAULT_MODE;
  const db = getDb();
  db.prepare(`
    INSERT INTO chat_settings (chat_id, mode, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      mode = excluded.mode,
      updated_at = excluded.updated_at
  `).run(chatId, mode);
  cache.mode.set(chatId, mode);
  logger.info({ chatId, mode }, 'DB set_mode');
}

function getTriggers(chatId) {
  if (cache.triggers.has(chatId)) {
    const raw = cache.triggers.get(chatId);
    return new Set(raw.split(',').filter((t) => VALID_TRIGGERS.has(t.trim().toLowerCase())).map((t) => t.trim().toLowerCase()));
  }
  const db = getDb();
  const row = db.prepare('SELECT triggers FROM chat_settings WHERE chat_id = ?').get(chatId);
  const raw = row?.triggers ?? DEFAULT_TRIGGERS;
  cache.triggers.set(chatId, raw);
  return new Set(raw.split(',').filter((t) => VALID_TRIGGERS.has(t.trim().toLowerCase())).map((t) => t.trim().toLowerCase()));
}

function setTriggers(chatId, triggers) {
  const valid = [...triggers].filter((t) => VALID_TRIGGERS.has(t));
  const raw = valid.sort().join(',') || '';
  const db = getDb();
  db.prepare(`
    INSERT INTO chat_settings (chat_id, triggers, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      triggers = excluded.triggers,
      updated_at = excluded.updated_at
  `).run(chatId, raw);
  cache.triggers.set(chatId, raw);
  logger.info({ chatId, triggers: raw }, 'DB set_triggers');
}

function clearSettings(chatId) {
  const db = getDb();
  db.prepare('DELETE FROM chat_settings WHERE chat_id = ?').run(chatId);
  cache.prompt.delete(chatId);
  cache.permission.delete(chatId);
  cache.mode.delete(chatId);
  cache.triggers.delete(chatId);
  logger.info({ chatId }, 'DB clear_settings');
}

function getStats(chatId, periodType, periodKey) {
  const db = getDb();
  const rows = db.prepare(
    'SELECT stat_key, stat_value FROM chat_stats WHERE chat_id = ? AND period_type = ? AND period_key = ?'
  ).all(chatId, periodType, periodKey);
  const result = {};
  for (const row of rows) {
    result[row.stat_key] = row.stat_value;
  }
  return result;
}

function getTopUsers(chatId, periodType, periodKey, limit = 5) {
  const db = getDb();
  const rows = db.prepare(
    `SELECT sender_ref, sender_name, invoke_count FROM chat_user_stats
     WHERE chat_id = ? AND period_type = ? AND period_key = ?
     ORDER BY invoke_count DESC LIMIT ?`
  ).all(chatId, periodType, periodKey, limit);
  return rows.map((row) => ({ senderRef: row.sender_ref, senderName: row.sender_name, invokeCount: row.invoke_count }));
}

const _llm2ModelCache = new Map();
const _defaultModelCache = null;

function getDefaultLlm2Model() {
  const db = getDb();
  const row = db.prepare(
    'SELECT model_id, display_name, description FROM llm_models WHERE is_active = 1 ORDER BY sort_order ASC LIMIT 1'
  ).get();
  if (row) {
    return { modelId: row.model_id, displayName: row.display_name, description: row.description };
  }
  return null;
}

function getLlm2Model(chatId) {
  if (_llm2ModelCache.has(chatId)) {
    return _llm2ModelCache.get(chatId);
  }
  const db = getDb();
  const row = db.prepare('SELECT llm2_model FROM chat_settings WHERE chat_id = ?').get(chatId);
  const value = row?.llm2_model ?? null;
  _llm2ModelCache.set(chatId, value);
  return value;
}

function setLlm2Model(chatId, modelId) {
  const db = getDb();
  db.prepare(`
    INSERT INTO chat_settings (chat_id, llm2_model, updated_at)
    VALUES (?, ?, datetime('now'))
    ON CONFLICT(chat_id) DO UPDATE SET
      llm2_model = excluded.llm2_model,
      updated_at = excluded.updated_at
  `).run(chatId, modelId);
  _llm2ModelCache.set(chatId, modelId);
  logger.info({ chatId, modelId }, 'DB set_llm2_model');
}

function getAllActiveModels() {
  const db = getDb();
  const rows = db.prepare(
    'SELECT model_id, display_name, description, sort_order FROM llm_models WHERE is_active = 1 ORDER BY sort_order ASC'
  ).all();
  return rows.map((row) => ({
    modelId: row.model_id,
    displayName: row.display_name,
    description: row.description,
    sortOrder: row.sort_order,
  }));
}

function getAllModels() {
  const db = getDb();
  const rows = db.prepare(
    'SELECT model_id, display_name, description, is_active, sort_order FROM llm_models ORDER BY sort_order ASC'
  ).all();
  return rows.map((row) => ({
    modelId: row.model_id,
    displayName: row.display_name,
    description: row.description,
    isActive: Boolean(row.is_active),
    sortOrder: row.sort_order,
  }));
}

function addModel(modelId, displayName, description = '', sortOrder = null) {
  const db = getDb();
  if (sortOrder === null) {
    const maxOrder = db.prepare('SELECT MAX(sort_order) as max_order FROM llm_models').get();
    sortOrder = (maxOrder?.max_order ?? -1) + 1;
  }
  try {
    db.prepare(
      'INSERT INTO llm_models (model_id, display_name, description, sort_order) VALUES (?, ?, ?, ?)'
    ).run(modelId, displayName, description, sortOrder);
    logger.info({ modelId, displayName }, 'DB add_model');
    return true;
  } catch (err) {
    if (err.code === 'SQLITE_CONSTRAINT_PRIMARYKEY') {
      return false;
    }
    throw err;
  }
}

function updateModel(modelId, { displayName, description, isActive, sortOrder } = {}) {
  const db = getDb();
  const existing = db.prepare('SELECT model_id FROM llm_models WHERE model_id = ?').get(modelId);
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
  db.prepare(`UPDATE llm_models SET ${updates.join(', ')} WHERE model_id = ?`).run(...values);
  logger.info({ modelId }, 'DB update_model');
  return true;
}

function deleteModel(modelId) {
  const db = getDb();
  const existing = db.prepare('SELECT model_id FROM llm_models WHERE model_id = ?').get(modelId);
  if (!existing) return false;
  db.prepare('DELETE FROM llm_models WHERE model_id = ?').run(modelId);
  logger.info({ modelId }, 'DB delete_model');
  return true;
}

export {
  getDb,
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
