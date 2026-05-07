import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import Database from 'better-sqlite3';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '../..');
const workerDir = path.join(rootDir, 'tests/db-stress');
const nodeWorkerPath = path.join(workerDir, 'node-worker.mjs');
const pythonWorkerPath = path.join(workerDir, 'python_worker.py');

const pythonBin = process.env.PYTHON || process.env.PYTHON_BIN || 'python3';
const nodeWorkers = Number(process.env.STRESS_NODE_WORKERS || '4');
const pythonWorkers = Number(process.env.STRESS_PYTHON_WORKERS || '4');
const iterations = Number(process.env.STRESS_ITERATIONS || '120');
const chatCount = Number(process.env.STRESS_CHAT_COUNT || '24');

function dbPaths(dataDir) {
  return {
    settings: path.join(dataDir, 'settings.db'),
    stats: path.join(dataDir, 'stats.db'),
    moderation: path.join(dataDir, 'moderation.db'),
  };
}

function stressEnv(dataDir, extra = {}) {
  const paths = dbPaths(dataDir);
  return {
    ...process.env,
    DATA_DIR: dataDir,
    SETTINGS_DB_PATH: paths.settings,
    STATS_DB_PATH: paths.stats,
    MODERATION_DB_PATH: paths.moderation,
    BOT_SETTINGS_DB_PATH: paths.settings,
    BOT_STATS_DB_PATH: paths.stats,
    BOT_MODERATION_DB_PATH: paths.moderation,
    LOG_LEVEL: process.env.LOG_LEVEL || 'silent',
    DB_BUSY_TIMEOUT_MS: process.env.DB_BUSY_TIMEOUT_MS || '30000',
    DB_BUSY_TIMEOUT_SECONDS: process.env.DB_BUSY_TIMEOUT_SECONDS || '30',
    DB_OPERATION_RETRY_MAX: process.env.DB_OPERATION_RETRY_MAX || '8',
    DB_OPERATION_RETRY_BASE_MS: process.env.DB_OPERATION_RETRY_BASE_MS || '50',
    DB_OPERATION_RETRY_BASE_SECONDS: process.env.DB_OPERATION_RETRY_BASE_SECONDS || '0.05',
    STRESS_ITERATIONS: String(iterations),
    STRESS_CHAT_COUNT: String(chatCount),
    ...extra,
  };
}

function pipeChild(child, label) {
  child.stdout.on('data', (data) => process.stdout.write(`[${label}] ${data}`));
  child.stderr.on('data', (data) => process.stderr.write(`[${label}] ${data}`));
}

function runChild(command, args, env, label) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: rootDir, env, stdio: ['ignore', 'pipe', 'pipe'] });
    pipeChild(child, label);
    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(`${label} failed with code=${code} signal=${signal}`));
    });
  });
}

async function initSchema(dataDir) {
  const env = stressEnv(dataDir);
  const dbModuleUrl = pathToFileURL(path.join(rootDir, 'src/db.js'));
  dbModuleUrl.search = `?stress=${Date.now()}-${Math.random()}`;
  Object.assign(process.env, env);
  const db = await import(dbModuleUrl.href);
  db.init();
  db.closeAllDbs();
}

async function runConcurrentWorkers(dataDir) {
  const jobs = [];
  for (let i = 0; i < nodeWorkers; i += 1) {
    jobs.push(
      runChild(
        process.execPath,
        [nodeWorkerPath],
        stressEnv(dataDir, { WORKER_ID: String(i) }),
        `node-${i}`,
      ),
    );
  }
  for (let i = 0; i < pythonWorkers; i += 1) {
    jobs.push(
      runChild(
        pythonBin,
        [pythonWorkerPath],
        stressEnv(dataDir, { WORKER_ID: String(i + nodeWorkers) }),
        `python-${i}`,
      ),
    );
  }
  await Promise.all(jobs);
}

function assertIntegrity(dbPath, tables) {
  const db = new Database(dbPath, { readonly: true, fileMustExist: true });
  try {
    const integrityRows = db.pragma('integrity_check');
    const allOk = integrityRows.every((row) => row.integrity_check === 'ok');
    if (!allOk) throw new Error(`${dbPath} failed integrity_check: ${JSON.stringify(integrityRows)}`);

    const journalMode = db.pragma('journal_mode', { simple: true });
    if (journalMode !== 'wal') throw new Error(`${dbPath} journal_mode=${journalMode}`);

    for (const table of tables) {
      const row = db.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get();
      if (!row || row.count <= 0) throw new Error(`${dbPath} has no rows in ${table}`);
    }
  } finally {
    db.close();
  }
}

function verifyStressData(dataDir) {
  const paths = dbPaths(dataDir);
  assertIntegrity(paths.settings, ['chat_settings', 'llm_models']);
  assertIntegrity(paths.stats, ['chat_stats', 'chat_user_stats']);
  assertIntegrity(paths.moderation, ['chat_mutes']);
}

async function runCorruptionRecoveryCheck() {
  const dataDir = await fs.mkdtemp(path.join(os.tmpdir(), 'wazzap-db-corrupt-'));
  const paths = dbPaths(dataDir);
  await fs.mkdir(dataDir, { recursive: true });
  await Promise.all(Object.values(paths).map((dbPath) => fs.writeFile(dbPath, 'not a sqlite database')));

  await Promise.all([
    runChild(process.execPath, [nodeWorkerPath], stressEnv(dataDir, { WORKER_ID: '1000', STRESS_ITERATIONS: '16' }), 'recovery-node'),
    runChild(pythonBin, [pythonWorkerPath], stressEnv(dataDir, { WORKER_ID: '2000', STRESS_ITERATIONS: '16' }), 'recovery-python'),
  ]);

  verifyStressData(dataDir);

  const backups = await Promise.all(
    Object.values(paths).map(async (dbPath) => {
      const basename = path.basename(dbPath);
      const files = await fs.readdir(path.dirname(dbPath));
      return files.some((file) => file.startsWith(`${basename}.corrupted`) && file.endsWith('.bak'));
    }),
  );
  if (!backups.every(Boolean)) throw new Error('corrupt database backups were not preserved for every DB');
}

const dataDir = process.env.DB_STRESS_DATA_DIR || await fs.mkdtemp(path.join(os.tmpdir(), 'wazzap-db-stress-'));
await fs.mkdir(dataDir, { recursive: true });
console.log(`database stress data dir: ${dataDir}`);
console.log(`workers: node=${nodeWorkers} python=${pythonWorkers} iterations=${iterations} chats=${chatCount}`);

await initSchema(dataDir);
await runConcurrentWorkers(dataDir);
verifyStressData(dataDir);
await runCorruptionRecoveryCheck();
console.log('database stress test passed');
