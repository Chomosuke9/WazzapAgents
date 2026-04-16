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
import { parseSlashCommand } from './commands.js';
import { handleCommandListener } from './commandHandler.js';
import { isOwnerJid } from '../participants.js';
import { roleFlagsForJid } from '../participants.js';
import { getCachedGroupMetadata, defaultGroupContext, getGroupContext, currentBotAliases } from '../groupContext.js';
import { normalizeJid } from '../identifiers.js';
import {
  getLlm2Model,
  setLlm2Model,
  getAllActiveModels,
  getDefaultLlm2Model,
  deleteModel,
} from '../db.js';
import { sendRichMessage, sendNativeFlow } from './interactive/index.js';

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

  async function handleButtonResponse(msg, chatId, senderId) {
    const buttonsResponse = msg?.message?.buttonsResponseMessage;
    const listResponse = msg?.message?.listResponseMessage;
    if (!buttonsResponse && !listResponse) return false;

    const selectedId = (buttonsResponse?.selectedButtonId) || (listResponse?.singleSelectReply?.selectedRowId);
    if (!selectedId) return false;

    const isGroup = chatId.endsWith('@g.us');
    const group = isGroup ? (getCachedGroupMetadata(chatId) || defaultGroupContext(chatId)) : null;
    const senderRole = isGroup ? roleFlagsForJid(group?.participantRoles, senderId) : { isAdmin: false, isSuperAdmin: false };
    const senderIsAdmin = senderRole.isAdmin || senderRole.isSuperAdmin;
    const senderIsOwner = isOwnerJid(senderId);

    try {
      if (selectedId.startsWith('model_select:')) {
        const modelId = selectedId.replace('model_select:', '');
        const canUse = isGroup ? senderIsAdmin : senderIsOwner;
        if (!canUse) {
          await sock.sendMessage(chatId, { text: 'Only group admins or bot owner can change the model.' });
          return true;
        }
        setLlm2Model(chatId, modelId);
        const models = getAllActiveModels();
        const model = models.find((m) => m.modelId === modelId);
        const displayName = model?.displayName || modelId;
        await sock.sendMessage(chatId, { text: `Model diubah ke: ${displayName}` });
        return true;
      }

      if (selectedId.startsWith('settings:')) {
        const action = selectedId.replace('settings:', '');
        const canUse = isGroup ? senderIsAdmin : senderIsOwner;
        if (!canUse) {
          await sock.sendMessage(chatId, { text: 'Only group admins or bot owner can access settings.' });
          return true;
        }
        if (action === 'model') {
          const models = getAllActiveModels();
          if (models.length === 0) {
            await sock.sendMessage(chatId, { text: 'No models available.' });
            return true;
          }
          const currentModelId = getLlm2Model(chatId);
          const defaultModel = getDefaultLlm2Model();
          const activeModelId = currentModelId || defaultModel?.modelId || null;
          const sections = models.map((m) => ({
            title: m.displayName,
            rows: [{
              title: m.displayName + (m.modelId === activeModelId ? ' ✓' : ''),
              description: m.description || '',
              id: `model_select:${m.modelId}`,
            }],
          }));
          await sendNativeFlow(sock, chatId, 'Pilih Model LLM', [
            {
              name: 'single_select',
              buttonParamsJson: JSON.stringify({ title: 'Pilih Model', sections }),
            },
          ], { footer: 'Model saat ini: ' + (activeModelId || 'default') });
          return true;
        }
        if (action === 'prompt') {
          await sock.sendMessage(chatId, { text: 'Gunakan /prompt <teks> untuk mengubah prompt.' });
          return true;
        }
        if (action === 'permission') {
          await sock.sendMessage(chatId, { text: 'Gunakan /permission <0-3> untuk mengubah level.' });
          return true;
        }
        return true;
      }

      if (selectedId.startsWith('modelcfg:')) {
        if (!isOwnerJid(senderId)) {
          await sock.sendMessage(chatId, { text: 'Only bot owner can manage models.' });
          return true;
        }
        const subcommand = selectedId.replace('modelcfg:', '');
        if (subcommand === 'list') {
          const models = getAllActiveModels();
          if (models.length === 0) {
            await sock.sendMessage(chatId, { text: 'No models configured.' });
            return true;
          }
          const lines = ['*Daftar Model:*'];
          const defaultModel = getDefaultLlm2Model();
          for (const m of models) {
            const isDefault = defaultModel?.modelId === m.modelId;
            lines.push(`${isDefault ? '✓' : '○'} ${m.displayName} (${m.modelId})`);
          }
          await sock.sendMessage(chatId, { text: lines.join('\n') });
          return true;
        }
        return true;
      }

      if (selectedId.startsWith('modelcfg_remove:')) {
        if (!isOwnerJid(senderId)) {
          await sock.sendMessage(chatId, { text: 'Only bot owner can manage models.' });
          return true;
        }
        const modelId = selectedId.replace('modelcfg_remove:', '');
        const sections = [
          {
            title: 'Konfirmasi Hapus',
            rows: [
              { title: 'Ya, Hapus', description: `Hapus model ${modelId}`, id: `modelcfg_confirm_remove:${modelId}` },
              { title: 'Batal', description: 'Batalkan operasi', id: 'modelcfg_cancel_remove' },
            ],
          },
        ];
        await sendNativeFlow(sock, chatId, `⚠️ Hapus "${modelId}"?`, [
          {
            name: 'single_select',
            buttonParamsJson: JSON.stringify({ title: 'Konfirmasi', sections }),
          },
        ], { footer: 'Tindakan ini tidak dapat dibatalkan' });
        return true;
      }

      if (selectedId === 'modelcfg_cancel_remove') {
        await sock.sendMessage(chatId, { text: 'Operasi dibatalkan.' });
        return true;
      }

      if (selectedId.startsWith('modelcfg_confirm_remove:')) {
        if (!isOwnerJid(senderId)) {
          await sock.sendMessage(chatId, { text: 'Only bot owner can manage models.' });
          return true;
        }
        const modelId = selectedId.replace('modelcfg_confirm_remove:', '');
        const success = deleteModel(modelId);
        await sock.sendMessage(chatId, { text: success ? `Model "${modelId}" dihapus.` : `Model "${modelId}" tidak ditemukan.` });
        return true;
      }
    } catch (err) {
      logger.error({ err }, 'button response handler error');
    }
    return false;
  }

  // Listener 1: Command handler (non-blocking, instant response)
  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify' || !Array.isArray(messages) || messages.length === 0) return;
    for (const msg of messages) {
      try {
        const chatId = msg?.key?.remoteJid;
        if (!chatId || chatId === 'status@broadcast') continue;
        if (!msg?.message) continue;
        if (msg?.key?.fromMe) continue;

        const fromId = msg.key.participant || msg.key.remoteJid;
        const senderId = normalizeJid(fromId) || fromId;

        if (await handleButtonResponse(msg, chatId, senderId)) {
          continue;
        }

        const text = msg.message?.conversation || msg.message?.extendedTextMessage?.text || null;
        if (!text || typeof text !== 'string') continue;

        const slashCommand = parseSlashCommand(text);
        if (!slashCommand) continue;

        const isGroup = chatId.endsWith('@g.us');
        const chatType = isGroup ? 'group' : 'private';

        let senderIsAdmin = false;
        let botIsAdmin = false;
        let botIsSuperAdmin = false;
        let group = null;

        if (isGroup) {
          group = getCachedGroupMetadata(chatId) || defaultGroupContext(chatId);
          const senderRole = roleFlagsForJid(group?.participantRoles, senderId);
          senderIsAdmin = senderRole.isAdmin || senderRole.isSuperAdmin;
          botIsAdmin = Boolean(group?.botIsAdmin);
          botIsSuperAdmin = Boolean(group?.botIsSuperAdmin);
        }

        const context = {
          slashCommand,
          chatId,
          chatType,
          senderId,
          senderIsAdmin,
          senderIsOwner: isOwnerJid(senderId),
          senderRole: isGroup ? roleFlagsForJid(group?.participantRoles, senderId) : { isAdmin: false, isSuperAdmin: false },
          senderDisplay: msg.pushName || '',
          botIsAdmin,
          botIsSuperAdmin,
          contextMsgId: msg.key.id,
          text,
          group,
          msg,
        };

        await handleCommandListener(msg, context);
      } catch (err) {
        logger.error({ err }, 'command listener error');
      }
    }
  });

  // Listener 2: Chatbot handler (send to Python)
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
