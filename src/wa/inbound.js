import logger from '../logger.js';
import config from '../config.js';
import wsClient from '../wsClient.js';
import {
  normalizeJid,
  normalizeContextMsgId,
  ensureContextMsgId,
  rememberSenderRef,
  rememberMessage,
} from '../identifiers.js';
import {
  rememberParticipantName,
  lookupParticipantName,
  roleFlagsForJid,
  fallbackParticipantLabel,
  compactParticipantJids,
  isOwnerJid,
} from '../participants.js';
import {
  getCachedGroupMetadata,
  defaultGroupContext,
  getGroupContext,
  normalizeGroupJoinAction,
  invalidateGroupMetadata,
  parseGroupJoinStub,
  getGroupParticipantName,
  currentBotAliases,
} from '../groupContext.js';
import {
  unwrapMessage,
  extractMentionedJids,
  extractLocationData,
  formatLocationText,
  extractText,
  extractQuoted,
} from '../messageParser.js';
import { saveMedia } from '../mediaHandler.js';
import { getSock } from './connection.js';
import { withTimeout, escapeRegex } from './utils.js';
import {
  resolveParticipantLabel,
  emitGroupJoinContextEvent,
} from './events.js';
import { parseSlashCommand, handleBroadcastCommand, handleInfoCommand, handleDebugCommand, handleJoinCommand } from './commands.js';

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

async function handleGroupParticipantsUpdate(update) {
  const sock = getSock();
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
  const sock = getSock();
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

  if (slashCommand && slashCommand.command === 'debug') {
    await handleDebugCommand({ chatId, senderId, args: slashCommand.args });
    return;
  }

  if (slashCommand && slashCommand.command === 'join') {
    await handleJoinCommand({ chatId, senderId, args: slashCommand.args });
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

export {
  buildMentionedParticipants,
  handleGroupParticipantsUpdate,
  handleIncomingMessage,
};
