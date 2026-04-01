import { spawn } from 'child_process';
import makeWASocket, {
  fetchLatestBaileysVersion,
  useMultiFileAuthState,
  DisconnectReason,
} from 'baileys';
import logger from '../logger.js';
import config from '../config.js';
import wsClient from '../wsClient.js';
import { setSockAccessor, invalidateGroupMetadata } from '../groupContext.js';
import { runWithConcurrency } from './utils.js';

let sock;

function getSock() {
  return sock;
}

function printQrInTerminal(qr) {
  try {
    const proc = spawn('qrencode', ['-t', 'ANSIUTF8', '-o', '-']);
    proc.stdin.write(qr);
    proc.stdin.end();
    proc.stdout.on('data', (chunk) => process.stdout.write(chunk.toString()));
    proc.stderr.on('data', (chunk) => logger.debug({ qrErr: chunk.toString() }, 'qrencode stderr'));
    proc.on('error', (err) => {
      logger.warn({ err }, 'qrencode not available; showing raw QR string');
      console.log('QR:', qr);
    });
  } catch (err) {
    logger.warn({ err }, 'failed to render QR; showing raw');
    console.log('QR:', qr);
  }
}

async function startWhatsApp() {
  // Lazy import to avoid circular dependency: inbound/events import getSock from connection,
  // and connection imports handlers from inbound/events at call time only.
  const { handleIncomingMessage, handleGroupParticipantsUpdate } = await import('./inbound.js');
  const { emitGroupJoinContextEvent } = await import('./events.js');
  const { ensureContextMsgId, messageIdIndexKey } = await import('../identifiers.js');
  const { GROUP_JOIN_STUB_TYPES } = await import('../caches.js');
  const { parseGroupJoinStub } = await import('../groupContext.js');

  const { state, saveCreds } = await useMultiFileAuthState(config.authDir);
  const { version } = await fetchLatestBaileysVersion();
  logger.info({ version }, 'starting whatsapp socket');

  sock = makeWASocket({
    version,
    auth: state,
    syncFullHistory: false,
    browser: ['WazzapAgents', 'Chrome', '1.0'],
    markOnlineOnConnect: false,
    defaultQueryTimeoutMs: config.sendTimeoutMs,
  });
  setSockAccessor(() => sock);

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('connection.update', (update) => {
    const { connection, lastDisconnect, qr } = update;
    if (qr) {
      logger.info('Scan QR to authenticate (valid for 20 seconds)');
      printQrInTerminal(qr);
    }
    if (connection === 'close') {
      const statusCode = lastDisconnect?.error?.output?.statusCode;
      const reason = lastDisconnect?.error;
      logger.warn({ statusCode, reason }, 'connection closed, reconnecting');
      wsClient.send({ type: 'whatsapp_status', payload: { status: 'closed', reason: statusCode, instanceId: config.instanceId } });
      if (statusCode !== DisconnectReason.loggedOut) {
        startWhatsApp().catch((err) => logger.error({ err }, 'reconnect failed'));
      } else {
        logger.error('Logged out from WhatsApp. Delete auth folder to re-pair.');
      }
    } else if (connection === 'open') {
      logger.info('WhatsApp socket connected');
      wsClient.send({ type: 'whatsapp_status', payload: { status: 'open', instanceId: config.instanceId } });
    }
  });

  sock.ev.on('groups.update', (updates) => {
    if (!Array.isArray(updates)) return;
    for (const update of updates) {
      const jid = update?.id;
      if (!jid) continue;
      invalidateGroupMetadata(jid);
    }
  });

  sock.ev.on('group-participants.update', async (update) => {
    try {
      await handleGroupParticipantsUpdate(update);
    } catch (err) {
      logger.error({ err, update }, 'failed handling group participants update');
    }
  });

  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (!Array.isArray(messages) || messages.length === 0) return;
    const batchStartMs = Date.now();
    const isNotify = type === 'notify';
    const precomputedContextByMessage = new Map();

    if (!isNotify) {
      await runWithConcurrency(messages, config.upsertConcurrency, async (msg) => {
        try {
          const stubEvent = parseGroupJoinStub(msg);
          if (stubEvent) {
            await emitGroupJoinContextEvent(stubEvent);
          }
        } catch (err) {
          logger.error({ err }, 'failed handling message');
        }
      });
    } else {
      const notifyGroups = new Map();
      for (const msg of messages) {
        const chatId = msg?.key?.remoteJid || '__unknown_chat__';
        const bucket = notifyGroups.get(chatId) || [];
        bucket.push(msg);
        notifyGroups.set(chatId, bucket);

        const messageId = msg?.key?.id;
        if (!chatId || !messageId || chatId === 'status@broadcast') continue;
        if (GROUP_JOIN_STUB_TYPES.has(msg?.messageStubType) || !msg?.message) continue;
        const contextMsgId = ensureContextMsgId(chatId, messageId);
        precomputedContextByMessage.set(messageIdIndexKey(chatId, messageId), contextMsgId);
      }

      const groupedMessages = Array.from(notifyGroups.values());
      await runWithConcurrency(groupedMessages, config.upsertConcurrency, async (groupMessages) => {
        for (const msg of groupMessages) {
          try {
            const chatId = msg?.key?.remoteJid;
            const messageId = msg?.key?.id;
            const precomputedContextMsgId = (chatId && messageId)
              ? precomputedContextByMessage.get(messageIdIndexKey(chatId, messageId))
              : null;
            await handleIncomingMessage(msg, { precomputedContextMsgId });
          } catch (err) {
            logger.error({ err }, 'failed handling message');
          }
        }
      });
    }

    const batchTotalMs = Date.now() - batchStartMs;
    if (config.perfLogEnabled && messages.length > 1 && batchTotalMs >= config.perfLogThresholdMs) {
      logger.info({
        type,
        messageCount: messages.length,
        upsertConcurrency: config.upsertConcurrency,
        chatGroups: isNotify ? new Set(messages.map((msg) => msg?.key?.remoteJid || '__unknown_chat__')).size : null,
        batchTotalMs,
      }, 'slow messages.upsert batch');
    }
  });

  return sock;
}

export {
  getSock,
  startWhatsApp,
};
