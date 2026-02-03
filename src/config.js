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

const config = {
  instanceId: process.env.INSTANCE_ID || 'default',
  wsEndpoint: process.env.LLM_WS_ENDPOINT,
  wsToken: process.env.LLM_WS_TOKEN || null,
  reconnectIntervalMs: Number(process.env.WS_RECONNECT_MS || 5000),
  authDir: AUTH_DIR,
  mediaDir: MEDIA_DIR,
  logLevel: process.env.LOG_LEVEL || 'info',
  downloadTimeoutMs: Number(process.env.DOWNLOAD_TIMEOUT_MS || 60000),
  sendTimeoutMs: Number(process.env.SEND_TIMEOUT_MS || 60000),
};

export default config;
