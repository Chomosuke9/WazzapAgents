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

fs.ensureDirSync(AUTH_DIR);
fs.ensureDirSync(MEDIA_DIR);

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

const config = {
  instanceId: process.env.INSTANCE_ID || 'default',
  wsEndpoint: process.env.LLM_WS_ENDPOINT,
  wsToken: process.env.LLM_WS_TOKEN || null,
  reconnectIntervalMs: positiveInt(process.env.WS_RECONNECT_MS, 5000),
  authDir: AUTH_DIR,
  mediaDir: MEDIA_DIR,
  logLevel: process.env.LOG_LEVEL || 'info',
  downloadTimeoutMs: positiveInt(process.env.DOWNLOAD_TIMEOUT_MS, 60000),
  sendTimeoutMs: positiveInt(process.env.SEND_TIMEOUT_MS, 60000),
  upsertConcurrency: positiveInt(process.env.UPSERT_CONCURRENCY, 2),
  perfLogEnabled: process.env.PERF_LOG_ENABLED !== '0',
  perfLogThresholdMs: nonNegativeInt(process.env.PERF_LOG_THRESHOLD_MS, 400),
};

export default config;
