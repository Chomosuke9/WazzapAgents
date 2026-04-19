import path from 'path';
import fs from 'fs-extra';
import { fileURLToPath } from 'url';
import { config as dotenvConfig } from 'dotenv';

dotenvConfig();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ROOT = path.resolve(__dirname, '..');
const DATA_DIR = process.env.DATA_DIR || path.join(ROOT, 'data');
const AUTH_DIR = path.join(DATA_DIR, 'auth');
const MEDIA_DIR = process.env.MEDIA_DIR || path.join(DATA_DIR, 'media');
const STICKERS_DIR = process.env.STICKERS_DIR || path.join(DATA_DIR, 'stickers');
const SETTINGS_DB_PATH = process.env.SETTINGS_DB_PATH || path.join(DATA_DIR, 'settings.db');
const STATS_DB_PATH = process.env.STATS_DB_PATH || path.join(DATA_DIR, 'stats.db');
const MODERATION_DB_PATH = process.env.MODERATION_DB_PATH || path.join(DATA_DIR, 'moderation.db');

fs.ensureDirSync(AUTH_DIR);
fs.ensureDirSync(MEDIA_DIR);
fs.ensureDirSync(STICKERS_DIR);

function positiveInt(value, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(1, Math.floor(parsed));
}

function nonNegativeInt(value, fallback) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(0, Math.floor(parsed));
}

function normalizeOwnerJid(raw) {
  const trimmed = raw.trim().toLowerCase();
  if (!trimmed) return null;
  if (trimmed.includes('@')) return trimmed;
  return `${trimmed}@s.whatsapp.net`;
}

function parseJidList(raw) {
  if (!raw || typeof raw !== 'string') return [];
  return raw
    .split(',')
    .map(normalizeOwnerJid)
    .filter(Boolean);
}

const config = {
  instanceId: process.env.INSTANCE_ID || 'default',
  wsEndpoint: process.env.LLM_WS_ENDPOINT,
  wsToken: process.env.LLM_WS_TOKEN || null,
  dataDir: DATA_DIR,
  settingsDbPath: SETTINGS_DB_PATH,
  statsDbPath: STATS_DB_PATH,
  moderationDbPath: MODERATION_DB_PATH,
  reconnectIntervalMs: positiveInt(process.env.WS_RECONNECT_MS, 5000),
  authDir: AUTH_DIR,
  mediaDir: MEDIA_DIR,
  stickersDir: STICKERS_DIR,
  logLevel: process.env.LOG_LEVEL || 'info',
  groupMetadataTimeoutMs: positiveInt(process.env.GROUP_METADATA_TIMEOUT_MS, 8000),
  downloadTimeoutMs: positiveInt(process.env.DOWNLOAD_TIMEOUT_MS, 60000),
  sendTimeoutMs: positiveInt(process.env.SEND_TIMEOUT_MS, 60000),
  upsertConcurrency: positiveInt(process.env.UPSERT_CONCURRENCY, 2),
  perfLogEnabled: process.env.PERF_LOG_ENABLED !== '0',
  perfLogThresholdMs: nonNegativeInt(process.env.PERF_LOG_THRESHOLD_MS, 400),
  botOwnerJids: parseJidList(process.env.BOT_OWNER_JIDS),
  llmReplyInteractive: process.env.LLM_REPLY_INTERACTIVE === 'true',
  llmReplyFooter: process.env.LLM_REPLY_FOOTER || '',
};

export default config;
