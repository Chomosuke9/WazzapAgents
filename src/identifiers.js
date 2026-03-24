import { createHash } from 'crypto';
import { jidNormalizedUser } from 'baileys';
import {
  messageCache,
  MAX_CACHE,
  MAX_KEY_INDEX,
  messageKeyIndex,
  messageIdToContextId,
  contextCounterByChat,
  senderRefRegistryByChat,
  cacheSetBounded,
} from './caches.js';

function normalizeJid(jid) {
  if (!jid || typeof jid !== 'string') return null;
  try {
    return jidNormalizedUser(jid);
  } catch {
    return jid;
  }
}

function normalizeContextMsgId(value) {
  if (typeof value !== 'string' && typeof value !== 'number') return null;
  const raw = String(value).trim();
  const match = raw.match(/^<?\s*(\d{6})\s*>?$/);
  if (!match) return null;
  return match[1];
}

function contextIndexKey(chatId, contextMsgId) {
  return `${chatId}::${contextMsgId}`;
}

function messageIdIndexKey(chatId, messageId) {
  return `${chatId}::${messageId}`;
}

function nextContextMsgId(chatId) {
  const current = contextCounterByChat.get(chatId) || 0;
  const bounded = current % 1_000_000;
  contextCounterByChat.set(chatId, (bounded + 1) % 1_000_000);
  return String(bounded).padStart(6, '0');
}

function ensureSenderRefRegistry(chatId) {
  let registry = senderRefRegistryByChat.get(chatId);
  if (!registry) {
    registry = {
      senderToRef: new Map(),
      refToSender: new Map(),
      senderToParticipant: new Map(),
    };
    senderRefRegistryByChat.set(chatId, registry);
  }
  return registry;
}

function makeSenderRef(chatId, senderId, attempt = 0) {
  const digest = createHash('sha1').update(`${chatId}|${senderId}|${attempt}`).digest('hex');
  const numeric = Number.parseInt(digest.slice(0, 12), 16);
  return numeric.toString(36).padStart(6, '0').slice(0, 6);
}

function isPhoneJid(jid) {
  return typeof jid === 'string'
    && (jid.endsWith('@s.whatsapp.net') || jid.endsWith('@c.us'));
}

function rememberSenderRef(chatId, senderId, participantJid = null) {
  if (!chatId || !senderId) return null;
  const canonicalSenderId = normalizeJid(senderId) || senderId;
  const canonicalParticipant = normalizeJid(participantJid) || participantJid || canonicalSenderId;
  const registry = ensureSenderRefRegistry(chatId);

  const existingRef = registry.senderToRef.get(canonicalSenderId);
  if (existingRef) {
    const existingParticipant = registry.senderToParticipant.get(canonicalSenderId);
    if (!isPhoneJid(existingParticipant) || isPhoneJid(canonicalParticipant)) {
      registry.senderToParticipant.set(canonicalSenderId, canonicalParticipant);
    }
    return existingRef;
  }

  for (let attempt = 0; attempt < 128; attempt += 1) {
    const candidate = makeSenderRef(chatId, canonicalSenderId, attempt);
    const owner = registry.refToSender.get(candidate);
    if (owner && owner !== canonicalSenderId) continue;
    registry.senderToRef.set(canonicalSenderId, candidate);
    registry.refToSender.set(candidate, canonicalSenderId);
    registry.senderToParticipant.set(canonicalSenderId, canonicalParticipant);
    return candidate;
  }

  const fallback = `${Date.now() % 1_000_000}`.padStart(6, '0');
  registry.senderToRef.set(canonicalSenderId, fallback);
  registry.refToSender.set(fallback, canonicalSenderId);
  registry.senderToParticipant.set(canonicalSenderId, canonicalParticipant);
  return fallback;
}

function resolveSenderByRef(chatId, senderRef) {
  if (!chatId || typeof senderRef !== 'string') return null;
  const registry = senderRefRegistryByChat.get(chatId);
  if (!registry) return null;
  return registry.refToSender.get(senderRef.trim().toLowerCase()) || null;
}

function resolveParticipantBySenderId(chatId, senderId) {
  if (!chatId || !senderId) return null;
  const registry = senderRefRegistryByChat.get(chatId);
  if (!registry) return null;
  return registry.senderToParticipant.get(senderId) || null;
}

function buildNormalizedMessageKey(rawKey, chatId, senderId = null, fromMe = false) {
  const keyId = rawKey?.id;
  const remoteJid = rawKey?.remoteJid || chatId;
  if (!keyId || !remoteJid) return null;
  const normalizedSenderId = normalizeJid(senderId) || senderId || null;
  const normalized = {
    id: keyId,
    remoteJid,
    participant: rawKey?.participant || normalizedSenderId || undefined,
    fromMe: Boolean(rawKey?.fromMe ?? fromMe),
  };
  return normalized;
}

function rememberMessageKeyIndex({
  chatId,
  contextMsgId,
  rawKey,
  senderId = null,
  senderRef = null,
  senderIsAdmin = false,
  fromMe = false,
  timestampMs = Date.now(),
}) {
  const normalizedContextMsgId = normalizeContextMsgId(contextMsgId);
  if (!chatId || !normalizedContextMsgId) return null;
  const key = buildNormalizedMessageKey(rawKey, chatId, senderId, fromMe);
  if (!key) return null;
  const normalizedSenderId = normalizeJid(senderId) || senderId || null;
  const entry = {
    contextMsgId: normalizedContextMsgId,
    id: key.id,
    chatId,
    remoteJid: key.remoteJid,
    participant: key.participant || null,
    fromMe: Boolean(key.fromMe),
    timestampMs: Number(timestampMs) || Date.now(),
    senderId: normalizedSenderId,
    senderRef: senderRef || null,
    senderIsAdmin: Boolean(senderIsAdmin),
    key: {
      id: key.id,
      remoteJid: key.remoteJid,
      participant: key.participant || undefined,
      fromMe: Boolean(key.fromMe),
    },
  };
  cacheSetBounded(
    messageKeyIndex,
    contextIndexKey(chatId, normalizedContextMsgId),
    entry,
    MAX_KEY_INDEX
  );
  cacheSetBounded(
    messageIdToContextId,
    messageIdIndexKey(chatId, key.id),
    normalizedContextMsgId,
    MAX_KEY_INDEX * 2
  );
  return entry;
}

function getIndexedMessageByContextId(chatId, contextMsgId) {
  const normalizedContextMsgId = normalizeContextMsgId(contextMsgId);
  if (!chatId || !normalizedContextMsgId) return null;
  return messageKeyIndex.get(contextIndexKey(chatId, normalizedContextMsgId)) || null;
}

function findContextMsgIdByMessageId(chatId, messageId) {
  if (!chatId || !messageId) return null;
  const found = messageIdToContextId.get(messageIdIndexKey(chatId, messageId));
  return normalizeContextMsgId(found);
}

function ensureContextMsgId(chatId, messageId) {
  const known = findContextMsgIdByMessageId(chatId, messageId);
  if (known) return known;
  return nextContextMsgId(chatId);
}

function rememberMessage(msg, {
  chatId = msg?.key?.remoteJid || null,
  contextMsgId = null,
  senderId = null,
  senderRef = null,
  senderIsAdmin = false,
  fromMe = false,
  timestampMs = Date.now(),
} = {}) {
  if (!msg?.key?.id) return null;
  messageCache.set(msg.key.id, msg);
  if (messageCache.size > MAX_CACHE) {
    const firstKey = messageCache.keys().next().value;
    messageCache.delete(firstKey);
  }
  if (!chatId) return null;
  const resolvedContextMsgId = normalizeContextMsgId(contextMsgId) || ensureContextMsgId(chatId, msg.key.id);
  rememberMessageKeyIndex({
    chatId,
    contextMsgId: resolvedContextMsgId,
    rawKey: msg.key,
    senderId,
    senderRef,
    senderIsAdmin,
    fromMe,
    timestampMs,
  });
  return resolvedContextMsgId;
}

function resolveQuotedMessage(chatId, target) {
  if (!target) return null;
  const maybeContext = normalizeContextMsgId(target);
  if (!maybeContext) {
    return messageCache.get(target) || null;
  }
  const entry = getIndexedMessageByContextId(chatId, maybeContext);
  if (!entry) return null;
  const cached = entry.id ? messageCache.get(entry.id) : null;
  if (cached) return cached;
  return { key: entry.key, message: { conversation: '' } };
}

function mentionHandleForJid(jid) {
  if (!jid || typeof jid !== 'string') return null;
  const normalized = normalizeJid(jid) || jid;
  const local = String(normalized).split('@')[0] || '';
  const cleaned = local.replace(/[^0-9A-Za-z._-]/g, '');
  if (!cleaned) return null;
  return `@${cleaned}`;
}

function resolveMentionTargetBySenderRef(chatId, senderRef) {
  if (!chatId || !senderRef) return null;
  const senderId = resolveSenderByRef(chatId, senderRef);
  if (!senderId) return null;
  const participantFromRegistry = resolveParticipantBySenderId(chatId, senderId);
  return normalizeJid(participantFromRegistry) || normalizeJid(senderId) || senderId || null;
}

export {
  normalizeJid,
  normalizeContextMsgId,
  contextIndexKey,
  messageIdIndexKey,
  nextContextMsgId,
  ensureSenderRefRegistry,
  makeSenderRef,
  isPhoneJid,
  rememberSenderRef,
  resolveSenderByRef,
  resolveParticipantBySenderId,
  buildNormalizedMessageKey,
  rememberMessageKeyIndex,
  getIndexedMessageByContextId,
  findContextMsgIdByMessageId,
  ensureContextMsgId,
  rememberMessage,
  resolveQuotedMessage,
  mentionHandleForJid,
  resolveMentionTargetBySenderRef,
};
