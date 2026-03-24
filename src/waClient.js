import path from 'path';
import { spawn } from 'child_process';
import makeWASocket, {
  fetchLatestBaileysVersion,
  useMultiFileAuthState,
  DisconnectReason,
} from 'baileys';
import logger from './logger.js';
import config from './config.js';
import wsClient from './wsClient.js';
import {
  messageCache,
  GROUP_JOIN_STUB_TYPES,
} from './caches.js';
import {
  normalizeJid,
  normalizeContextMsgId,
  messageIdIndexKey,
  nextContextMsgId,
  isPhoneJid,
  rememberSenderRef,
  resolveSenderByRef,
  resolveParticipantBySenderId,
  rememberMessageKeyIndex,
  getIndexedMessageByContextId,
  ensureContextMsgId,
  rememberMessage,
  resolveQuotedMessage,
  mentionHandleForJid,
  resolveMentionTargetBySenderRef,
} from './identifiers.js';
import {
  compactParticipantJids,
  rememberParticipantName,
  lookupParticipantName,
  roleFlagsForJid,
  fallbackParticipantLabel,
  normalizeKickTargets,
  isOwnerJid,
} from './participants.js';
import {
  setSockAccessor,
  currentBotAliases,
  defaultGroupContext,
  getCachedGroupMetadata,
  invalidateGroupMetadata,
  getGroupContext,
  getGroupParticipantName,
  normalizeGroupJoinAction,
  parseGroupJoinStub,
  dedupeGroupJoinEvent,
} from './groupContext.js';
import {
  unwrapMessage,
  extractMentionedJids,
  extractLocationData,
  formatLocationText,
  extractText,
  extractQuoted,
} from './messageParser.js';
import {
  resolveAllowedAttachmentPath,
  saveMedia,
} from './mediaHandler.js';

let sock;

async function runWithConcurrency(items, concurrency, worker) {
  if (!Array.isArray(items) || items.length === 0) return;
  const limit = Math.max(1, Number(concurrency) || 1);
  let cursor = 0;

  async function consume() {
    while (cursor < items.length) {
      const idx = cursor;
      cursor += 1;
      await worker(items[idx], idx);
    }
  }

  const workers = [];
  const workerCount = Math.min(limit, items.length);
  for (let i = 0; i < workerCount; i += 1) {
    workers.push(consume());
  }
  await Promise.all(workers);
}

async function withTimeout(promise, timeoutMs, label = 'operation') {
  const timeout = Number(timeoutMs);
  if (!Number.isFinite(timeout) || timeout <= 0) return promise;

  let timer = null;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => {
          reject(actionError('timeout', `${label} timed out`, `timeout after ${timeout}ms`));
        }, timeout);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

function escapeRegex(value) {
  if (typeof value !== 'string') return '';
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
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


async function resolveParticipantLabel(chatId, participantJid) {
  const normalized = normalizeJid(participantJid) || participantJid;
  if (!normalized) return 'unknown';
  const candidates = Array.from(new Set([participantJid, normalized].filter(Boolean)));
  for (const candidate of candidates) {
    const fromCache = lookupParticipantName(candidate);
    if (fromCache) return fromCache;
  }
  if (chatId?.endsWith('@g.us')) {
    for (const candidate of candidates) {
      const fromGroup = await getGroupParticipantName(chatId, candidate);
      if (fromGroup) return fromGroup;
    }
  }
  return fallbackParticipantLabel(normalized);
}

async function buildMentionedParticipants(chatId, mentionedJids, botAliasSet = null) {
  if (!Array.isArray(mentionedJids) || mentionedJids.length === 0) return null;
  const normalizedMentions = Array.from(new Set(
    mentionedJids
      .map((jid) => normalizeJid(jid) || jid)
      .filter(Boolean)
  ));
  if (normalizedMentions.length === 0) return null;

  const rows = [];
  for (const participantJid of normalizedMentions) {
    const normalized = normalizeJid(participantJid) || participantJid;
    if (!normalized) continue;
    const name = await resolveParticipantLabel(chatId, normalized);
    const senderRef = rememberSenderRef(chatId, normalized, normalized) || null;
    const isBot = Boolean(
      botAliasSet instanceof Set
      && (botAliasSet.has(normalized) || botAliasSet.has(participantJid))
    );
    rows.push({
      jid: normalized,
      senderRef,
      name: name || fallbackParticipantLabel(normalized),
      isBot,
    });
  }
  return rows.length > 0 ? rows : null;
}

function makeEventMessageId(prefix) {
  const stamp = Date.now();
  const rand = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${stamp}_${rand}`;
}

async function emitGroupJoinContextEvent({
  chatId,
  action,
  participants,
  actorId = null,
  timestampMs = Date.now(),
  messageId = null,
  messageKey = null,
  source = 'group-participants.update',
}) {
  const normalizedParticipants = compactParticipantJids(participants);
  if (!chatId || !chatId.endsWith('@g.us') || normalizedParticipants.length === 0) return;
  if (!dedupeGroupJoinEvent(chatId, normalizedParticipants, action, timestampMs)) return;

  const group = await getGroupContext(chatId, { forceRefresh: true });
  const labels = await Promise.all(
    normalizedParticipants.map((participantJid) => resolveParticipantLabel(chatId, participantJid))
  );
  const mentionedParticipants = normalizedParticipants.map((participantJid, idx) => {
    const senderRef = rememberSenderRef(chatId, participantJid, participantJid) || null;
    const name = labels[idx] || fallbackParticipantLabel(participantJid);
    return {
      jid: participantJid,
      senderRef,
      name,
      isBot: false,
    };
  });
  const uniqueParticipantLabels = Array.from(new Set(
    mentionedParticipants
      .map((item) => {
        const senderRef = typeof item.senderRef === 'string' ? item.senderRef.trim() : '';
        if (senderRef) return `${item.name} (${senderRef})`;
        return item.name;
      })
      .filter(Boolean)
  ));
  const normalizedActorId = normalizeJid(actorId) || null;
  const actorName = normalizedActorId
    ? await resolveParticipantLabel(chatId, normalizedActorId)
    : null;
  const actorSenderId = normalizedActorId || 'group-system@wazzap.local';
  const actorSenderRef = rememberSenderRef(chatId, actorSenderId, actorSenderId) || 'unknown';
  const actorRole = roleFlagsForJid(group?.participantRoles, actorSenderId);
  const hasAnchorKey = Boolean(messageKey?.id);
  const contextMsgId = hasAnchorKey ? nextContextMsgId(chatId) : null;
  const normalizedTimestampMs = Number(timestampMs) || Date.now();
  const resolvedMessageId = messageId || makeEventMessageId('group_join');

  if (contextMsgId) {
    rememberMessageKeyIndex({
      chatId,
      contextMsgId,
      rawKey: messageKey,
      senderId: actorSenderId,
      senderRef: actorSenderRef,
      senderIsAdmin: actorRole.isAdmin || actorRole.isSuperAdmin,
      fromMe: false,
      timestampMs: normalizedTimestampMs,
    });
  }

  const joinedText = uniqueParticipantLabels.length === 1
    ? `${uniqueParticipantLabels[0]} joined the group.`
    : `New members joined the group: ${uniqueParticipantLabels.join(', ')}.`;
  const byText = actorName ? ` Added by ${actorName}.` : '';
  const text = `Group update: ${joinedText}${byText}`;

  const payload = {
    messageId: resolvedMessageId,
    instanceId: config.instanceId,
    chatId,
    chatName: group.name || chatId,
    chatType: 'group',
    senderId: actorSenderId,
    senderRef: actorSenderRef,
    senderName: actorName || 'Group System',
    senderIsAdmin: actorRole.isAdmin || actorRole.isSuperAdmin,
    isGroup: true,
    botIsAdmin: Boolean(group?.botIsAdmin),
    botIsSuperAdmin: Boolean(group?.botIsSuperAdmin),
    fromMe: false,
    timestampMs: normalizedTimestampMs,
    messageType: 'groupParticipantsUpdate',
    text,
    quoted: null,
    attachments: [],
    mentionedJids: normalizedParticipants,
    mentionedParticipants,
    location: null,
    contextOnly: true,
    triggerLlm1: true,
    groupDescription: group.description,
    groupPromptOveride: group.promptOveride,
    groupEvent: {
      action: action || 'join',
      participants: normalizedParticipants,
      actorId: normalizedActorId,
      actorName,
      source,
    },
  };
  if (contextMsgId) payload.contextMsgId = contextMsgId;

  wsClient.send({ type: 'incoming_message', payload });
}

function emitBotActionContextEvent({
  chatId,
  action,
  text,
  result = null,
}) {
  if (!sock || !chatId || !text) return;

  const isGroup = chatId.endsWith('@g.us');
  const group = isGroup
    ? (getCachedGroupMetadata(chatId) || defaultGroupContext(chatId))
    : null;
  const senderId = normalizeJid(sock.user?.id) || 'bot@wazzap.local';
  const senderRef = rememberSenderRef(chatId, senderId, senderId) || 'unknown';
  const payload = {
    messageId: makeEventMessageId('action_log'),
    instanceId: config.instanceId,
    chatId,
    chatName: isGroup ? (group?.name || chatId) : chatId,
    chatType: isGroup ? 'group' : 'private',
    senderId,
    senderRef,
    senderName: sock.user?.name || 'LLM',
    senderIsAdmin: Boolean(group?.botIsAdmin),
    isGroup,
    botIsAdmin: Boolean(group?.botIsAdmin),
    botIsSuperAdmin: Boolean(group?.botIsSuperAdmin),
    fromMe: true,
    contextOnly: true,
    triggerLlm1: false,
    timestampMs: Date.now(),
    messageType: 'actionLog',
    text,
    quoted: null,
    attachments: [],
    mentionedJids: null,
    mentionedParticipants: null,
    location: null,
    groupDescription: group?.description || null,
    groupPromptOveride: group?.promptOveride || null,
    actionLog: {
      action,
      result,
    },
  };

  wsClient.send({ type: 'incoming_message', payload });
}

async function handleGroupParticipantsUpdate(update) {
  if (!sock) return;
  const chatId = update?.id;
  if (!chatId || !chatId.endsWith('@g.us')) return;
  invalidateGroupMetadata(chatId);

  const action = normalizeGroupJoinAction(update?.action);
  const joinActions = new Set(['add', 'invite', 'join', 'approve']);
  if (!joinActions.has(action)) return;

  const participants = compactParticipantJids(Array.isArray(update?.participants) ? update.participants : []);
  if (participants.length === 0) return;
  const actorId = compactParticipantJids([update?.authorPn, update?.author])[0] || null;
  await emitGroupJoinContextEvent({
    chatId,
    action,
    participants,
    actorId,
    timestampMs: Date.now(),
    source: 'group-participants.update',
  });
}

async function handleIncomingMessage(msg, { precomputedContextMsgId = null } = {}) {
  if (!sock) return;
  const perfStartMs = Date.now();
  const perf = {
    groupMs: 0,
    quotedMs: 0,
    mediaMs: 0,
  };

  const stubEvent = parseGroupJoinStub(msg);
  if (stubEvent) {
    await emitGroupJoinContextEvent(stubEvent);
    return;
  }

  if (!msg.message) return;
  const remoteJid = msg.key.remoteJid;
  if (!remoteJid || remoteJid === 'status@broadcast') return;

  const chatId = remoteJid;
  const isGroup = chatId.endsWith('@g.us');
  const chatType = isGroup ? 'group' : 'private';
  const fromMe = Boolean(msg.key?.fromMe);
  const selfJid = normalizeJid(sock.user?.id) || null;
  const botAliases = new Set(
    currentBotAliases()
      .map((jid) => normalizeJid(jid) || jid)
      .filter(Boolean)
  );
  if (selfJid) botAliases.add(selfJid);
  const fromId = msg.key.participant || (fromMe ? selfJid : msg.key.remoteJid);
  const senderId = normalizeJid(fromId) || fromId || normalizeJid(msg.key.remoteJid) || msg.key.remoteJid;
  const senderDisplay = msg.pushName || lookupParticipantName(senderId) || senderId;
  rememberParticipantName(fromId, msg.pushName || '');
  rememberParticipantName(senderId, senderDisplay);

  const groupStartMs = Date.now();
  const group = isGroup
    ? (
      fromMe
        ? (getCachedGroupMetadata(chatId) || defaultGroupContext(chatId))
        : await getGroupContext(chatId)
    )
    : null;
  perf.groupMs = Date.now() - groupStartMs;
  const senderRole = isGroup ? roleFlagsForJid(group?.participantRoles, senderId) : { isAdmin: false, isSuperAdmin: false };
  const senderRef = rememberSenderRef(chatId, senderId, msg.key.participant || senderId) || 'unknown';
  const contextMsgId = normalizeContextMsgId(precomputedContextMsgId) || ensureContextMsgId(chatId, msg.key.id);
  const chatName = isGroup ? (group?.name || chatId) : chatId;

  const { contentType, message: innerMessage } = unwrapMessage(msg.message);
  if (!contentType || !innerMessage) return;
  const content = innerMessage[contentType];
  const location = extractLocationData(innerMessage);
  const locationText = location ? formatLocationText(location) : null;
  const baseText = extractText(innerMessage);
  const text = [baseText, locationText].filter(Boolean).join('\n') || null;
  const quotedStartMs = Date.now();
  const quoted = await extractQuoted(innerMessage, chatId, { allowGroupLookup: !fromMe, getGroupParticipantName });
  perf.quotedMs = Date.now() - quotedStartMs;
  const mentionedJidsRaw = extractMentionedJids(innerMessage);
  const mentionedJids = Array.isArray(mentionedJidsRaw)
    ? Array.from(new Set(
      mentionedJidsRaw
        .map((jid) => normalizeJid(jid) || jid)
        .filter(Boolean)
    ))
    : null;
  const mentionedParticipants = Array.isArray(mentionedJids) && mentionedJids.length > 0
    ? await buildMentionedParticipants(chatId, mentionedJids, botAliases)
    : null;
  const botMentionedByJid = Boolean(
    Array.isArray(mentionedJids)
    && mentionedJids.some((jid) => botAliases.has(normalizeJid(jid) || jid))
  );
  const botMentionTokens = Array.from(botAliases)
    .map((jid) => String(jid).split('@')[0]?.trim())
    .filter((token) => typeof token === 'string' && token.length >= 5);
  const botMentionedByText = Boolean(
    typeof text === 'string'
    && botMentionTokens.some((token) => (
      new RegExp(`(^|[^0-9A-Za-z_])@${escapeRegex(token)}(?=$|[^0-9A-Za-z_])`).test(text)
    ))
  );
  const botMentioned = botMentionedByJid || botMentionedByText;
  const quotedSenderId = normalizeJid(quoted?.senderId) || quoted?.senderId || null;
  const repliedToBot = Boolean(quotedSenderId && botAliases.has(quotedSenderId));

  const attachments = [];
  const mediaKinds = [
    'imageMessage',
    'videoMessage',
    'audioMessage',
    'documentMessage',
    'stickerMessage',
  ];
  if (mediaKinds.includes(contentType)) {
    const mediaStartMs = Date.now();
    try {
      const mediaInfo = await saveMedia(contentType, content, msg.key.id, withTimeout);
      if (mediaInfo) attachments.push(mediaInfo);
    } catch (err) {
      logger.error({ err }, 'failed saving media');
    } finally {
      perf.mediaMs = Date.now() - mediaStartMs;
    }
  }

  // Detect slash commands
  const slashCommand = (!fromMe && typeof text === 'string')
    ? parseSlashCommand(text)
    : null;

  // Handle /broadcast entirely on the gateway side
  if (slashCommand && slashCommand.command === 'broadcast') {
    await handleBroadcastCommand({
      chatId,
      senderId,
      text: slashCommand.args,
      quotedMessageId: quoted?.messageId || null,
      contextMsgId,
      msg,
    });
    return;
  }

  if (slashCommand && slashCommand.command === 'info') {
    await handleInfoCommand({
      chatId,
      senderId,
      senderDisplay,
      senderRole,
      isGroup,
      group,
    });
    return;
  }

  const payload = {
    contextMsgId,
    messageId: msg.key.id,
    instanceId: config.instanceId,
    chatId,
    chatName,
    chatType,
    senderId,
    senderRef,
    senderName: fromMe ? (senderDisplay || 'LLM') : senderDisplay,
    senderIsAdmin: senderRole.isAdmin || senderRole.isSuperAdmin,
    senderIsOwner: isOwnerJid(senderId),
    isGroup,
    botIsAdmin: Boolean(group?.botIsAdmin),
    botIsSuperAdmin: Boolean(group?.botIsSuperAdmin),
    fromMe,
    contextOnly: fromMe || contentType === 'reactionMessage',
    triggerLlm1: false,
    timestampMs: Number(msg.messageTimestamp) * 1000,
    messageType: contentType,
    text,
    quoted,
    attachments,
    mentionedJids,
    mentionedParticipants,
    botMentioned,
    repliedToBot,
    location,
    groupDescription: group?.description || null,
    groupPromptOveride: group?.promptOveride || null,
    slashCommand: slashCommand || null,
  };

  wsClient.send({ type: 'incoming_message', payload });
  rememberMessage(msg, {
    chatId,
    contextMsgId,
    senderId,
    senderRef,
    senderIsAdmin: payload.senderIsAdmin,
    fromMe,
    timestampMs: payload.timestampMs,
  });

  const totalMs = Date.now() - perfStartMs;
  if (config.perfLogEnabled && totalMs >= config.perfLogThresholdMs) {
    logger.info({
      chatId,
      messageId: msg.key.id,
      messageType: contentType,
      totalMs,
      groupMs: perf.groupMs,
      quotedMs: perf.quotedMs,
      mediaMs: perf.mediaMs,
      attachmentCount: attachments.length,
      isGroup,
      fromMe,
    }, 'slow inbound message processing');
  }
}

function actionError(code, message, detail = null) {
  const err = new Error(message);
  err.code = code;
  if (detail) err.detail = detail;
  return err;
}

async function reactToMessage({ chatId, contextMsgId, emoji }) {
  if (!sock) throw actionError('send_failed', 'WhatsApp socket not ready');
  if (typeof emoji !== 'string' || !emoji.trim()) {
    throw actionError('invalid_target', 'missing or empty emoji');
  }
  const normalizedContextMsgId = normalizeContextMsgId(contextMsgId);
  if (!normalizedContextMsgId) {
    throw actionError('invalid_target', 'invalid contextMsgId');
  }
  const indexed = getIndexedMessageByContextId(chatId, normalizedContextMsgId);
  if (!indexed) {
    throw actionError('not_found', 'context message not found');
  }
  if (indexed.chatId !== chatId) {
    throw actionError('invalid_target', 'context message belongs to a different chat');
  }
  try {
    await sock.sendMessage(chatId, {
      react: {
        text: emoji.trim(),
        key: indexed.key,
      },
    });
    emitBotActionContextEvent({
      chatId,
      action: 'react_message',
      text: `Action log: reacted ${emoji.trim()} to message <${normalizedContextMsgId}>.`,
      result: {
        contextMsgId: normalizedContextMsgId,
        emoji: emoji.trim(),
      },
    });
    return {
      contextMsgId: normalizedContextMsgId,
      emoji: emoji.trim(),
    };
  } catch (err) {
    throw actionError('send_failed', err?.message || 'failed to react to message');
  }
}

async function deleteMessageByContextId({ chatId, contextMsgId }) {
  if (!sock) throw actionError('send_failed', 'WhatsApp socket not ready');
  const normalizedContextMsgId = normalizeContextMsgId(contextMsgId);
  if (!normalizedContextMsgId) {
    throw actionError('invalid_target', 'invalid contextMsgId');
  }
  const indexed = getIndexedMessageByContextId(chatId, normalizedContextMsgId);
  if (!indexed) {
    throw actionError('not_found', 'context message not found');
  }
  if (indexed.chatId !== chatId) {
    throw actionError('invalid_target', 'context message belongs to a different chat');
  }
  try {
    await sock.sendMessage(chatId, { delete: indexed.key });
    emitBotActionContextEvent({
      chatId,
      action: 'delete_message',
      text: `Action log: deleted message <${normalizedContextMsgId}>.`,
      result: {
        contextMsgId: normalizedContextMsgId,
        messageId: indexed.id || null,
      },
    });
    return {
      contextMsgId: normalizedContextMsgId,
      messageId: indexed.id,
    };
  } catch (err) {
    throw actionError('send_failed', err?.message || 'failed to delete message');
  }
}

function parseParticipantUpdateStatus(rawStatus) {
  const status = Number(rawStatus);
  if (Number.isFinite(status)) return status;
  return 0;
}

async function maybeEmitKickAnchorReplies(chatId, successTargets) {
  if (!Array.isArray(successTargets) || successTargets.length === 0) return;
  const botSenderId = normalizeJid(sock?.user?.id) || 'bot@wazzap.local';
  const botSenderRef = rememberSenderRef(chatId, botSenderId, botSenderId) || 'unknown';
  const group = chatId.endsWith('@g.us') ? await getGroupContext(chatId) : null;

  for (const target of successTargets) {
    const quoted = resolveQuotedMessage(chatId, target.anchorContextMsgId);
    const text = `Moderation: removed ${target.senderRef}.`;
    try {
      const sent = await sock.sendMessage(chatId, { text }, quoted ? { quoted } : {});
      const contextMsgId = nextContextMsgId(chatId);
      rememberMessage(sent, {
        chatId,
        contextMsgId,
        senderId: botSenderId,
        senderRef: botSenderRef,
        senderIsAdmin: Boolean(group?.botIsAdmin),
        fromMe: true,
        timestampMs: Date.now(),
      });
    } catch (err) {
      logger.warn({ err, chatId, target }, 'failed sending autoReplyAnchor log');
    }
  }
}

async function kickMembers({
  chatId,
  targets = [],
  mode = 'partial_success',
  autoReplyAnchor = false,
}) {
  if (!sock) throw actionError('send_failed', 'WhatsApp socket not ready');
  if (!chatId || !chatId.endsWith('@g.us')) {
    throw actionError('not_group', 'kick_member can only run in group chats');
  }

  const group = await getGroupContext(chatId, { forceRefresh: true });
  if (!group?.botIsAdmin) {
    throw actionError('permission_denied', 'bot is not admin');
  }

  const selfAliases = new Set(currentBotAliases());
  const normalizedTargets = normalizeKickTargets(targets);
  const resolvedTargets = [];
  const results = [];

  for (const target of normalizedTargets) {
    const { senderRef, anchorContextMsgId } = target;
    if (!senderRef || !anchorContextMsgId) {
      results.push({
        senderRef: senderRef || null,
        anchorContextMsgId: anchorContextMsgId || null,
        ok: false,
        error: 'invalid_target',
        detail: 'senderRef or anchorContextMsgId invalid',
      });
      continue;
    }

    const senderId = resolveSenderByRef(chatId, senderRef);
    if (!senderId) {
      results.push({
        senderRef,
        anchorContextMsgId,
        ok: false,
        error: 'invalid_target',
        detail: 'unknown senderRef',
      });
      continue;
    }

    const anchor = getIndexedMessageByContextId(chatId, anchorContextMsgId);
    if (!anchor) {
      results.push({
        senderRef,
        anchorContextMsgId,
        ok: false,
        error: 'not_found',
        detail: 'anchor message not found',
      });
      continue;
    }
    if ((anchor.senderRef || '').toLowerCase() !== senderRef) {
      results.push({
        senderRef,
        anchorContextMsgId,
        ok: false,
        error: 'invalid_target',
        detail: 'anchor does not belong to senderRef',
      });
      continue;
    }

    const participantFromRegistry = resolveParticipantBySenderId(chatId, senderId);
    const participantJid = normalizeJid(participantFromRegistry) || normalizeJid(senderId) || senderId;
    if (!group.participantRoles?.[participantJid]) {
      results.push({
        senderRef,
        anchorContextMsgId,
        ok: false,
        error: 'invalid_target',
        detail: 'target is not an active group participant',
      });
      continue;
    }

    if (selfAliases.has(participantJid)) {
      results.push({
        senderRef,
        anchorContextMsgId,
        ok: false,
        error: 'invalid_target',
        detail: 'cannot kick bot/self',
      });
      continue;
    }

    const targetRole = roleFlagsForJid(group.participantRoles, participantJid);
    if (targetRole.isAdmin || targetRole.isSuperAdmin) {
      results.push({
        senderRef,
        anchorContextMsgId,
        ok: false,
        error: 'permission_denied',
        detail: 'cannot kick admin/superadmin',
      });
      continue;
    }

    resolvedTargets.push({
      senderRef,
      senderId: normalizeJid(senderId) || senderId,
      participantJid,
      anchorContextMsgId,
    });
  }

  const uniqueResolved = [];
  const seenParticipant = new Set();
  for (const target of resolvedTargets) {
    if (seenParticipant.has(target.participantJid)) continue;
    seenParticipant.add(target.participantJid);
    uniqueResolved.push(target);
  }

  if (uniqueResolved.length > 0) {
    let updateResponse;
    try {
      updateResponse = await sock.groupParticipantsUpdate(
        chatId,
        uniqueResolved.map((item) => item.participantJid),
        'remove'
      );
    } catch (err) {
      for (const target of uniqueResolved) {
        results.push({
          senderRef: target.senderRef,
          anchorContextMsgId: target.anchorContextMsgId,
          ok: false,
          error: 'send_failed',
          detail: err?.message || 'failed to execute kick',
        });
      }
      return { ok: false, mode, results };
    }

    const statusByParticipant = new Map();
    if (Array.isArray(updateResponse)) {
      for (const item of updateResponse) {
        const participantJid = normalizeJid(item?.jid || item?.id || item?.participant || item?.user);
        if (!participantJid) continue;
        statusByParticipant.set(participantJid, parseParticipantUpdateStatus(item?.status ?? item?.code));
      }
    }

    const successTargets = [];
    for (const target of uniqueResolved) {
      const status = statusByParticipant.has(target.participantJid)
        ? statusByParticipant.get(target.participantJid)
        : 200;
      const ok = status >= 200 && status < 300;
      if (ok) {
        successTargets.push(target);
      }
      results.push({
        senderRef: target.senderRef,
        anchorContextMsgId: target.anchorContextMsgId,
        ok,
        error: ok ? null : 'send_failed',
        detail: ok ? 'removed' : `remove_failed_status_${status}`,
      });
    }

    if (autoReplyAnchor && successTargets.length > 0) {
      await maybeEmitKickAnchorReplies(chatId, successTargets);
    }
    if (successTargets.length > 0) {
      const kickedRefs = successTargets.map(
        (item) => `${item.senderRef}@${item.anchorContextMsgId}`
      );
      const text = kickedRefs.length === 1
        ? `Action log: kicked ${kickedRefs[0]}.`
        : `Action log: kicked ${kickedRefs.length} members (${kickedRefs.join(', ')}).`;
      emitBotActionContextEvent({
        chatId,
        action: 'kick_member',
        text,
        result: {
          mode,
          targets: successTargets.map((item) => ({
            senderRef: item.senderRef,
            anchorContextMsgId: item.anchorContextMsgId,
            participantJid: item.participantJid,
          })),
        },
      });
    }
  }

  return {
    ok: results.some((item) => item.ok),
    mode,
    results,
  };
}

async function startWhatsApp() {
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

async function renderOutboundMentions(chatId, rawText, groupContext = null) {
  if (typeof rawText !== 'string' || !rawText.includes('@<')) {
    return { text: rawText, mentions: [], groupContext };
  }
  const tokens = Array.from(rawText.matchAll(/@<([^<>\r\n]+)>/g));
  if (tokens.length === 0) {
    return { text: rawText, mentions: [], groupContext };
  }

  let resolvedGroup = groupContext;
  let retried = false;
  let cursor = 0;
  let rendered = '';
  const mentionSet = new Set();

  for (const token of tokens) {
    const fullToken = token[0];
    const rawValue = typeof token[1] === 'string' ? token[1].trim() : '';
    const normalizedValue = rawValue.toLowerCase();
    const index = Number.isInteger(token.index) ? token.index : -1;
    if (index < 0) continue;

    rendered += rawText.slice(cursor, index);
    let replacement = rawValue ? `@${rawValue}` : '@';

    if (normalizedValue === 'all') {
      if (chatId?.endsWith('@g.us')) {
        let participants = Array.isArray(resolvedGroup?.participants) ? resolvedGroup.participants : [];
        if (participants.length === 0) {
          resolvedGroup = await getGroupContext(chatId, { forceRefresh: true });
          participants = Array.isArray(resolvedGroup?.participants) ? resolvedGroup.participants : [];
        }
        for (const participantJid of participants) {
          const normalizedParticipant = normalizeJid(participantJid) || participantJid;
          if (!normalizedParticipant) continue;
          mentionSet.add(normalizedParticipant);
        }
      }
      replacement = '@all';
    } else if (normalizedValue) {
      let participantJid = resolveMentionTargetBySenderRef(chatId, normalizedValue);
      if (!participantJid && !retried && chatId?.endsWith('@g.us')) {
        logger.debug({ chatId, senderRef: normalizedValue }, 'senderRef not found — force-refreshing group metadata');
        resolvedGroup = await getGroupContext(chatId, { forceRefresh: true });
        retried = true;
        participantJid = resolveMentionTargetBySenderRef(chatId, normalizedValue);
      }
      if (participantJid) {
        const normalizedParticipant = normalizeJid(participantJid) || participantJid;
        mentionSet.add(normalizedParticipant);
        replacement = mentionHandleForJid(normalizedParticipant) || replacement;
      } else {
        logger.warn({ chatId, senderRef: normalizedValue }, 'outbound mention resolution failed — token will render as plain text');
      }
    }

    rendered += replacement;
    cursor = index + fullToken.length;
  }

  rendered += rawText.slice(cursor);
  const mentionsArray = Array.from(mentionSet);
  for (const jid of mentionsArray) {
    if (!isPhoneJid(jid)) {
      logger.warn({ chatId, jid }, 'outbound mention contains non-phone JID — may not render as clickable');
    }
  }
  return {
    text: rendered,
    mentions: mentionsArray,
    groupContext: resolvedGroup,
  };
}

async function sendOutgoing({ chatId, text, attachments = [], replyTo }) {
  if (!sock) throw actionError('send_failed', 'WhatsApp socket not ready');
  if (!chatId) throw actionError('invalid_target', 'chatId is required');
  if (attachments != null && !Array.isArray(attachments)) {
    throw actionError('invalid_target', 'attachments must be an array');
  }

  const quoted = replyTo ? resolveQuotedMessage(chatId, replyTo) : null;
  if (replyTo && !quoted) {
    throw actionError('not_found', 'reply target not found');
  }

  const isGroup = chatId.endsWith('@g.us');
  let group = isGroup ? await getGroupContext(chatId) : null;
  const botSenderId = normalizeJid(sock.user?.id) || 'bot@wazzap.local';
  const botSenderRef = rememberSenderRef(chatId, botSenderId, botSenderId) || 'unknown';
  const normalizedText = typeof text === 'string' ? text.trim() : '';
  const normalizedAttachments = Array.isArray(attachments) ? attachments : [];
  if (!normalizedText && normalizedAttachments.length === 0) {
    throw actionError('invalid_target', 'send_message requires non-empty text or at least one attachment');
  }
  const sent = [];

  // send attachments first (with caption if provided)
  for (const att of normalizedAttachments) {
    const kindToken = typeof att?.kind === 'string' ? att.kind : (typeof att?.type === 'string' ? att.type : 'document');
    const kind = kindToken.trim().toLowerCase() || 'document';
    const filePath = await resolveAllowedAttachmentPath(att?.path, actionError);
    const content = {};
    if (kind === 'image') content.image = { url: filePath };
    else if (kind === 'video') content.video = { url: filePath };
    else if (kind === 'audio') content.audio = { url: filePath, ptt: false };
    else if (kind === 'sticker') content.sticker = { url: filePath };
    else content.document = { url: filePath, fileName: att.fileName || path.basename(filePath) };

    if (att.caption) {
      const renderedCaption = await renderOutboundMentions(chatId, String(att.caption), group);
      content.caption = renderedCaption.text;
      if (renderedCaption.mentions.length > 0) {
        content.mentions = renderedCaption.mentions;
      }
      group = renderedCaption.groupContext || group;
    }

    const sentMsg = await sock.sendMessage(chatId, content, quoted ? { quoted } : {});
    const contextMsgId = nextContextMsgId(chatId);
    rememberMessage(sentMsg, {
      chatId,
      contextMsgId,
      senderId: botSenderId,
      senderRef: botSenderRef,
      senderIsAdmin: Boolean(group?.botIsAdmin),
      fromMe: true,
      timestampMs: Date.now(),
    });
    sent.push({
      kind,
      contextMsgId,
      messageId: sentMsg?.key?.id || null,
    });
  }

  if (normalizedText) {
    const renderedText = await renderOutboundMentions(chatId, normalizedText, group);
    group = renderedText.groupContext || group;
    const textPayload = { text: renderedText.text };
    if (renderedText.mentions.length > 0) {
      textPayload.mentions = renderedText.mentions;
    }
    const sentMsg = await sock.sendMessage(chatId, textPayload, quoted ? { quoted } : {});
    const contextMsgId = nextContextMsgId(chatId);
    rememberMessage(sentMsg, {
      chatId,
      contextMsgId,
      senderId: botSenderId,
      senderRef: botSenderRef,
      senderIsAdmin: Boolean(group?.botIsAdmin),
      fromMe: true,
      timestampMs: Date.now(),
    });
    sent.push({
      kind: 'text',
      contextMsgId,
      messageId: sentMsg?.key?.id || null,
    });
  }
  if (sent.length === 0) {
    throw actionError('invalid_target', 'send_message produced no deliverable content');
  }

  return {
    sent,
    replyTo: normalizeContextMsgId(replyTo),
  };
}

// ---------------------------------------------------------------------------
// Slash command parsing
// ---------------------------------------------------------------------------

const SLASH_CMD_RE = /^\/(broadcast|prompt|reset|permission|info)\b\s*([\s\S]*)/i;

function parseSlashCommand(text) {
  if (!text || typeof text !== 'string') return null;
  const m = text.trim().match(SLASH_CMD_RE);
  if (!m) return null;
  return {
    command: m[1].toLowerCase(),
    args: (m[2] || '').trim(),
  };
}


async function handleBroadcastCommand({ chatId, senderId, text, quotedMessageId, contextMsgId, msg }) {
  if (!isOwnerJid(senderId)) {
    logger.info({ senderId, chatId }, '/broadcast rejected: not owner');
    try {
      await sock.sendMessage(chatId, { text: 'Only bot owners can use /broadcast.' });
    } catch (err) {
      logger.warn({ err }, 'failed sending broadcast rejection');
    }
    return;
  }

  // Collect all groups where bot is present
  let groupJids = [];
  try {
    const groups = await sock.groupFetchAllParticipating();
    groupJids = Object.keys(groups || {});
  } catch (err) {
    logger.error({ err }, 'failed fetching groups for broadcast');
    try {
      await sock.sendMessage(chatId, { text: 'Failed to fetch group list.' });
    } catch (e) { /* ignore */ }
    return;
  }

  if (groupJids.length === 0) {
    try {
      await sock.sendMessage(chatId, { text: 'Bot is not in any groups.' });
    } catch (e) { /* ignore */ }
    return;
  }

  let sent = 0;
  let failed = 0;

  if (text) {
    // Text broadcast: /broadcast <text>
    for (const groupJid of groupJids) {
      try {
        await sock.sendMessage(groupJid, { text });
        sent += 1;
      } catch (err) {
        logger.warn({ err, groupJid }, 'broadcast send failed');
        failed += 1;
      }
    }
  } else if (quotedMessageId) {
    // Forward broadcast: /broadcast (replying to a message)
    const cachedMsg = messageCache.get(quotedMessageId);
    if (!cachedMsg) {
      try {
        await sock.sendMessage(chatId, { text: 'Replied message not found in cache. Try replying to a more recent message.' });
      } catch (e) { /* ignore */ }
      return;
    }

    for (const groupJid of groupJids) {
      try {
        await sock.sendMessage(groupJid, { forward: cachedMsg });
        sent += 1;
      } catch (err) {
        logger.warn({ err, groupJid }, 'broadcast forward failed');
        failed += 1;
      }
    }
  } else {
    try {
      await sock.sendMessage(chatId, { text: 'Usage: /broadcast <text> or reply to a message with /broadcast.' });
    } catch (e) { /* ignore */ }
    return;
  }

  // Send confirmation
  try {
    const summary = `Broadcast complete: ${sent} group${sent !== 1 ? 's' : ''} sent${failed > 0 ? `, ${failed} failed` : ''}.`;
    await sock.sendMessage(chatId, { text: summary });
  } catch (err) {
    logger.warn({ err }, 'failed sending broadcast confirmation');
  }

  logger.info({ sent, failed, total: groupJids.length, chatId, senderId }, 'broadcast completed');
}

function truncateText(value, maxChars = 300) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.length <= maxChars) return trimmed;
  return `${trimmed.slice(0, Math.max(0, maxChars - 3))}...`;
}

async function handleInfoCommand({ chatId, senderId, senderDisplay, senderRole, isGroup, group }) {
  const isOwner = isOwnerJid(senderId);
  const roleLabel = isOwner
    ? 'owner'
    : (senderRole?.isSuperAdmin ? 'superadmin' : (senderRole?.isAdmin ? 'admin' : 'member'));
  const lines = [
    'Info pengguna:',
    `Nama: ${senderDisplay || 'unknown'}`,
    `JID: ${senderId || 'unknown'}`,
    `Peran: ${roleLabel}`,
    `Owner bot: ${isOwner ? 'ya' : 'tidak'}`,
  ];

  if (isGroup) {
    const groupName = group?.name || chatId;
    const memberCount = Array.isArray(group?.participants) ? group.participants.length : null;
    const description = truncateText(group?.description, 300);
    lines.push('');
    lines.push('Info grup:');
    lines.push(`Nama grup: ${groupName || 'unknown'}`);
    lines.push(`ID grup: ${chatId || 'unknown'}`);
    lines.push(`Jumlah anggota: ${typeof memberCount === 'number' ? memberCount : 'unknown'}`);
    lines.push(`Bot admin: ${group?.botIsAdmin ? 'ya' : 'tidak'}`);
    lines.push(`Bot superadmin: ${group?.botIsSuperAdmin ? 'ya' : 'tidak'}`);
    if (description) lines.push(`Deskripsi: ${description}`);
  } else {
    lines.push('');
    lines.push('Info chat:');
    lines.push('Tipe: private');
    lines.push(`ID chat: ${chatId || 'unknown'}`);
  }

  try {
    await sock.sendMessage(chatId, { text: lines.join('\n') });
  } catch (err) {
    logger.warn({ err, chatId }, 'failed sending /info response');
  }
}

// ---------------------------------------------------------------------------
// Read receipt and typing presence
// ---------------------------------------------------------------------------

async function markChatRead({ chatId, messageId, participant }) {
  if (!sock) return;
  try {
    const key = {
      remoteJid: chatId,
      id: messageId,
    };
    if (participant) key.participant = participant;
    await sock.readMessages([key]);
  } catch (err) {
    logger.warn({ err, chatId, messageId }, 'markChatRead failed');
  }
}

async function sendPresence({ chatId, type }) {
  if (!sock) return;
  try {
    // type: 'composing' | 'paused' | 'recording'
    await sock.sendPresenceUpdate(type || 'composing', chatId);
  } catch (err) {
    logger.warn({ err, chatId, type }, 'sendPresence failed');
  }
}

export {
  withTimeout,
  startWhatsApp,
  sendOutgoing,
  reactToMessage,
  deleteMessageByContextId,
  kickMembers,
  markChatRead,
  sendPresence,
};
