/**
 * inbound.js — Transform WhatsApp messages into normalized payloads for the Python bridge.
 *
 * This module handles the critical step between Baileys' raw message events and the
 * structured `incoming_message` payload that the Python bridge processes:
 *
 *   1. Parse group join stubs (emit as synthetic context events instead of forwarding)
 *   2. Normalize sender identity: JID → senderRef, resolve display names
 *   3. Determine bot-mention and replied-to-bot signals for LLM1 routing
 *   4. Download and validate media attachments (image/video/audio/document/sticker)
 *   5. Extract quoted (replied-to) messages with full metadata
 *   6. Build the final payload with all context needed by the LLM pipeline
 *
 * Bot messages (fromMe=true) are forwarded with `contextOnly=true` and
 * `triggerLlm1=false` so they enrich context without causing response loops.
 */
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
  emitBotRoleChangeEvent,
} from './events.js';
import { parseSlashCommand } from './command/index.js';

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

  const rawAction = typeof update?.action === 'string' ? update.action.toLowerCase() : '';
  const participants = compactParticipantJids(Array.isArray(update?.participants) ? update.participants : []);
  if (participants.length === 0) return;
  const actorId = compactParticipantJids([update?.authorPn, update?.author])[0] || null;

  // Handle promote/demote: invalidate AFTER event emission so that
  // emitBotRoleChangeEvent can read cached group name/description
  const roleActions = new Set(['promote', 'demote']);
  if (roleActions.has(rawAction)) {
    const botAliases = new Set(currentBotAliases());
    const botAffected = participants.some((p) => botAliases.has(normalizeJid(p) || p));
    if (botAffected) {
      emitBotRoleChangeEvent({
        chatId,
        action: rawAction,
        actorId,
      });
    }
    invalidateGroupMetadata(chatId);
    return;
  }

  // Handle join events: invalidate before (emitGroupJoinContextEvent
  // already forceRefreshes via getGroupContext)
  invalidateGroupMetadata(chatId);
  const action = normalizeGroupJoinAction(rawAction);
  const joinActions = new Set(['add', 'invite', 'join', 'approve']);
  if (!joinActions.has(action)) return;

  await emitGroupJoinContextEvent({
    chatId,
    action,
    participants,
    actorId,
    timestampMs: Date.now(),
    source: 'group-participants.update',
  });
}

/**
 * Handle a single incoming WhatsApp message.
 *
 * Builds a normalized `incoming_message` payload and sends it to the Python
 * bridge via `wsClient.send()`. Key behaviors:
 *
 *   - Bot's own messages are sent with `contextOnly=true`, `triggerLlm1=false`
 *   - Reaction messages are sent as `contextOnly=true` (no need for LLM response)
 *   - Interactive message replies are marked `contextOnly=true` (already handled by button handler)
 *   - Slash commands are detected and included in the payload for context enrichment,
 *     but command execution is handled in connection.js before this runs
 *
 * Performance: Logs slow processing if total time exceeds PERF_LOG_THRESHOLD_MS.
 *
 * @param {object} msg - Raw Baileys message object
 * @param {{precomputedContextMsgId?: string}} [options] - Pre-computed contextMsgId if known
 */
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
    ? await getGroupContext(chatId)
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
  const replyToInteractive = repliedToBot && quoted?.type === 'interactiveMessage';

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

  // Detect slash commands for context
  // Note: Commands are now handled by commandHandler.js in connection.js
  // We still detect slash commands and send to Python for context/history
  const slashCommand = (typeof text === 'string')
    ? parseSlashCommand(text)
    : null;

  // Mark if command was handled by Node.js (for Python to skip processing)
  const commandHandled = slashCommand ? true : false;

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
    senderIsSuperAdmin: Boolean(senderRole.isSuperAdmin),
    senderIsOwner: isOwnerJid(senderId),
    isGroup,
    botIsAdmin: Boolean(group?.botIsAdmin),
    botIsSuperAdmin: Boolean(group?.botIsSuperAdmin),
    fromMe,
    contextOnly: fromMe || contentType === 'reactionMessage' || replyToInteractive,
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
    commandHandled,
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
