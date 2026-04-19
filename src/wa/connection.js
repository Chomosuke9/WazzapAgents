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
  getAllModels,
  getDefaultLlm2Model,
  deleteModel,
  updateModel,
} from '../db.js';
import { sendRichMessage, sendNativeFlow } from './interactive/index.js';

const pendingForms = new Map();

function clearPendingForm(chatId) {
  pendingForms.delete(chatId);
}

function parseModelReply(chatId, text) {
  const form = pendingForms.get(chatId);
  if (!form) return null;

  if (form.type === 'edit_model') {
    const modelId = form.modelId;
    clearPendingForm(chatId);

    const updates = {};
    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);

    for (const line of lines) {
      const [key, ...valueParts] = line.split('=');
      if (!key || valueParts.length === 0) continue;
      const value = valueParts.join('=').trim();
      const k = key.trim().toLowerCase();

      if (k === 'name') updates.displayName = value;
      else if (k === 'desc') updates.description = value;
      else if (k === 'active') updates.isActive = value === '1' || value === 'true' || value === 'yes';
      else if (k === 'order') {
        const n = parseInt(value, 10);
        if (!isNaN(n)) updates.sortOrder = n;
      }
    }

    const success = updateModel(modelId, updates);
    return { action: 'edit_model', modelId, success, updates };
  }

  if (form.type === 'add_model') {
    clearPendingForm(chatId);
    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length < 2) {
      return { action: 'add_model', error: 'Format: model_id\\ndisplay_name\\n[description]' };
    }
    const [modelId, displayName, ...descParts] = lines;
    const description = descParts.join(' ');
    return { action: 'add_model', modelId, displayName, description };
  }

  return null;
}

async function showModelSelectionForEdit(sock, chatId) {
  const models = getAllModels();
  if (models.length === 0) {
    await sock.sendMessage(chatId, { text: 'No models to edit.' });
    return;
  }
  const sections = [
    {
      title: 'Select Model to Edit',
      rows: models.map((m) => ({
        id: `/modelcfg edit ${m.modelId}`,
        title: m.displayName + (m.isActive ? '' : ' (inactive)'),
        description: m.description || `ID: ${m.modelId}`,
      })),
    },
  ];
  await sendNativeFlow(sock, chatId, 'Edit Model', [
    {
      name: 'single_select',
      buttonParamsJson: JSON.stringify({ title: 'Select Model', sections }),
    },
  ], { footer: 'Select a model to edit' });
}

async function showModelEditForm(sock, chatId, senderId, modelId) {
  const models = getAllModels();
  const model = models.find((m) => m.modelId === modelId);
  if (!model) {
    await sock.sendMessage(chatId, { text: `Model "${modelId}" not found.` });
    return;
  }

  pendingForms.set(chatId, { type: 'edit_model', modelId, senderId });

  const helpText = `Edit Model: ${model.displayName}

Current values:
- name=${model.displayName}
- desc=${model.description || ''}
- active=${model.isActive ? '1' : '0'}
- order=${model.sortOrder}

Send your changes in format:
name=new_name
desc=new_description
active=1
order=2

Or send "cancel" to cancel.`;

  await sock.sendMessage(chatId, { text: helpText });
}

async function showModelAddForm(sock, chatId, senderId) {
  pendingForms.set(chatId, { type: 'add_model', senderId });

  const helpText = `Add New Model

Send the following format:
model_id
display_name
description (optional)

Example:
gpt-4o
GPT-4 Omni
Fast and capable model

Or send "cancel" to cancel.`;

  await sock.sendMessage(chatId, { text: helpText });
}

async function showModelSelectionForDefault(sock, chatId) {
  const models = getAllModels().filter((m) => m.isActive);
  if (models.length === 0) {
    await sock.sendMessage(chatId, { text: 'No active models to set as default.' });
    return;
  }
  const sections = [
    {
      title: 'Select Default Model',
      rows: models.map((m) => ({
        id: `/modelcfg default ${m.modelId}`,
        title: m.displayName,
        description: m.description || `ID: ${m.modelId}`,
      })),
    },
  ];
  await sendNativeFlow(sock, chatId, 'Set Default Model', [
    {
      name: 'single_select',
      buttonParamsJson: JSON.stringify({ title: 'Select Default', sections }),
    },
  ], { footer: 'Model with smallest order will be used as default' });
}

async function setDefaultModel(sock, chatId, modelId) {
  const models = getAllModels();
  const model = models.find((m) => m.modelId === modelId);
  if (!model) {
    await sock.sendMessage(chatId, { text: `Model "${modelId}" not found.` });
    return;
  }
  const allModels = getAllModels();
  const minOrder = Math.min(...allModels.map((m) => m.sortOrder));
  updateModel(modelId, { sortOrder: minOrder - 1 });
  wsClient.sendReliable({ type: 'invalidate_default_model' });
  await sock.sendMessage(chatId, { text: `Model "${model.displayName}" set as default.` });
}

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
      wsClient.sendReliable({ type: 'whatsapp_status', payload: { status: 'closed', reason: statusCode, instanceId: config.instanceId } });
      if (statusCode !== DisconnectReason.loggedOut) {
        startWhatsApp().catch((err) => logger.error({ err }, 'reconnect failed'));
      } else {
        logger.error('Logged out from WhatsApp. Delete auth folder to re-pair.');
      }
    } else if (connection === 'open') {
      logger.info('WhatsApp socket connected');
      wsClient.sendReliable({ type: 'whatsapp_status', payload: { status: 'open', instanceId: config.instanceId } });
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
    const interactiveResponse = msg?.message?.interactiveResponseMessage;
    
    const nativeFlowParams = (() => {
      try {
        const paramsStr = interactiveResponse?.nativeFlowResponseMessage?.paramsJson;
        if (paramsStr) return JSON.parse(paramsStr);
      } catch {}
      return null;
    })();
    const selectedId = (buttonsResponse?.selectedButtonId) || 
                       (listResponse?.singleSelectReply?.selectedRowId) ||
                       (nativeFlowParams?.id);
    
    if (!selectedId) return false;
    logger.info({ selectedId, chatId }, 'button selected');

    const isGroup = chatId.endsWith('@g.us');
    const group = isGroup ? (getCachedGroupMetadata(chatId) || defaultGroupContext(chatId)) : null;
    const senderRole = isGroup ? roleFlagsForJid(group?.participantRoles, senderId) : { isAdmin: false, isSuperAdmin: false };
    const senderIsAdmin = senderRole.isAdmin || senderRole.isSuperAdmin;
    const senderIsOwner = isOwnerJid(senderId);

    try {
      if (selectedId.startsWith('/')) {
        logger.info({ selectedId, chatId, senderId }, 'button click -> slash command');
        const { handleCommandListener } = await import('./commandHandler.js');
        const slashCommand = parseSlashCommand(selectedId);
        if (slashCommand) {
          const fakeMsg = {
            key: { ...msg.key, id: `btn_${Date.now()}` },
            message: { conversation: selectedId },
            pushName: msg.pushName,
          };
          const context = {
            slashCommand,
            chatId,
            chatType: isGroup ? 'group' : 'private',
            senderId,
            senderIsAdmin,
            senderIsOwner,
            senderRole,
            senderDisplay: msg.pushName || '',
            botIsAdmin: group?.botIsAdmin || false,
            botIsSuperAdmin: group?.botIsSuperAdmin || false,
            contextMsgId: null,
            text: selectedId,
            group,
            msg: fakeMsg,
          };
          await handleCommandListener(fakeMsg, context);
        }
        return true;
      }

      if (selectedId.startsWith('model_select:')) {
        const modelId = selectedId.replace('model_select:', '');
        const canUse = senderIsOwner || (isGroup && senderIsAdmin);
        if (!canUse) {
          await sock.sendMessage(chatId, { text: 'Only group admins or bot owner can change the model.' });
          return true;
        }
        setLlm2Model(chatId, modelId);
        wsClient.sendReliable({ type: 'set_llm2_model', chatId, modelId });
        wsClient.sendReliable({ type: 'invalidate_llm2_model', chatId });
        const models = getAllActiveModels();
        const model = models.find((m) => m.modelId === modelId);
        const displayName = model?.displayName || modelId;
        await sock.sendMessage(chatId, { text: `Model diubah ke: ${displayName}` });
        return true;
      }

      if (selectedId.startsWith('settings:')) {
        const action = selectedId.replace('settings:', '');
        const canUse = senderIsOwner || (isGroup && senderIsAdmin);
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

      if (selectedId.startsWith('modelcfg:') || selectedId.startsWith('modelcfg_')) {
        if (!isOwnerJid(senderId)) {
          await sock.sendMessage(chatId, { text: 'Only bot owner can manage models.' });
          return true;
        }
        
        const subcommand = selectedId.replace('modelcfg:', '').replace('modelcfg_', '');
        const colonIdx = subcommand.indexOf(':');
        const action = colonIdx >= 0 ? subcommand.slice(0, colonIdx) : subcommand;
        const modelId = colonIdx >= 0 ? subcommand.slice(colonIdx + 1) : '';

        if (action === 'list') {
          const models = getAllModels();
          if (models.length === 0) {
            await sock.sendMessage(chatId, { text: 'No models configured.' });
            return true;
          }
          const lines = ['*Daftar Model:*'];
          const defaultModel = getDefaultLlm2Model();
          for (const m of models) {
            const isDefault = defaultModel?.modelId === m.modelId;
            lines.push(`${isDefault ? '✓' : '○'} ${m.displayName} (${m.modelId})${isDefault ? ' [DEFAULT]' : ''}`);
            if (m.description) lines.push(`   ${m.description}`);
          }
          await sock.sendMessage(chatId, { text: lines.join('\n') });
          return true;
        }

        if (action === 'add') {
          await showModelAddForm(sock, chatId, senderId);
          return true;
        }

        if (action === 'edit') {
          await showModelSelectionForEdit(sock, chatId);
          return true;
        }

        if (action === 'default') {
          if (modelId) {
            await setDefaultModel(sock, chatId, modelId);
          } else {
            await showModelSelectionForDefault(sock, chatId);
          }
          return true;
        }

        if (action === 'remove') {
          if (modelId) {
            const success = deleteModel(modelId);
            const models = getAllModels();
            const model = models.find((m) => m.modelId === modelId);
            const displayName = model?.displayName || modelId;
            await sock.sendMessage(chatId, { text: success ? `Model "${displayName}" removed.` : `Model "${modelId}" not found.` });
          } else {
            await sock.sendMessage(chatId, { text: 'Usage: /modelcfg remove <model_id>' });
          }
          return true;
        }

        return true;
      }
    } catch (err) {
      logger.error({ err }, 'button response handler error');
    }
    return false;
  }

  // Listener 1: Command handler (non-blocking, instant response)
  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    logger.debug({ type, messageCount: messages?.length }, 'messages.upsert received');
    
    if (!Array.isArray(messages) || messages.length === 0) return;
    
    for (const msg of messages) {
      try {
        const chatId = msg?.key?.remoteJid;
        if (!chatId || chatId === 'status@broadcast') continue;
        if (!msg?.message) continue;
        if (msg?.key?.fromMe) continue;

        const fromId = msg.key.participant || msg.key.remoteJid;
        const senderId = normalizeJid(fromId) || fromId;

        logger.info({ chatId, senderId, msgKey: msg?.key?.id, type, msgContentType: msg.message ? Object.keys(msg.message).join(',') : 'none' }, 'message received');

        if (await handleButtonResponse(msg, chatId, senderId)) {
          continue;
        }

        const text = msg.message?.conversation || msg.message?.extendedTextMessage?.text || null;

        if (pendingForms.has(chatId) && senderId === pendingForms.get(chatId).senderId) {
          const normalizedText = text?.trim().toLowerCase();
          if (normalizedText === 'cancel' || normalizedText === 'batal') {
            clearPendingForm(chatId);
            await sock.sendMessage(chatId, { text: 'Operasi dibatalkan.' });
            continue;
          }

          const result = parseModelReply(chatId, text);
          if (result) {
            if (result.action === 'edit_model') {
              await sock.sendMessage(chatId, {
                text: result.success
                  ? `Model "${result.modelId}" diupdate.`
                  : `Model "${result.modelId}" tidak ditemukan.`
              });
            } else if (result.action === 'add_model') {
              if (result.error) {
                await sock.sendMessage(chatId, { text: result.error });
              } else {
                const { addModel } = await import('../db.js');
                const success = addModel(result.modelId, result.displayName, result.description);
                await sock.sendMessage(chatId, {
                  text: success
                    ? `Model "${result.displayName}" ditambahkan.`
                    : `Model "${result.modelId}" sudah ada.`
                });
              }
            }
            continue;
          }
        }
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
          group = await getGroupContext(chatId);
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
