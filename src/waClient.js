import path from 'path';
import fs from 'fs-extra';
import { spawn } from 'child_process';
import { createHash } from 'crypto';
import makeWASocket, {
  fetchLatestBaileysVersion,
  useMultiFileAuthState,
  DisconnectReason,
  getContentType,
  downloadContentFromMessage,
  jidNormalizedUser,
  normalizeMessageContent,
  WAMessageStubType,
} from 'baileys';
import logger from './logger.js';
import config from './config.js';
import { streamToFile } from './utils.js';
import wsClient from './wsClient.js';

let sock;
const messageCache = new Map(); // simple in-memory store for quoting outbound
const MAX_CACHE = 400;
const MAX_KEY_INDEX = 12_000;
const GROUP_METADATA_TTL_MS = 60_000;
const GROUP_JOIN_DEDUP_TTL_MS = 15_000;
const groupMetadataCache = new Map();
const participantNameCache = new Map();
const groupParticipantNameCache = new Map();
const groupJoinDedupCache = new Map();
const messageKeyIndex = new Map();
const messageIdToContextId = new Map();
const contextCounterByChat = new Map();
const senderRefRegistryByChat = new Map();
const GROUP_JOIN_STUB_TYPES = new Set([
  WAMessageStubType.GROUP_PARTICIPANT_ADD,
  WAMessageStubType.GROUP_PARTICIPANT_INVITE,
  WAMessageStubType.GROUP_PARTICIPANT_ADD_REQUEST_JOIN,
  WAMessageStubType.GROUP_PARTICIPANT_ACCEPT,
  WAMessageStubType.GROUP_PARTICIPANT_LINKED_GROUP_JOIN,
  WAMessageStubType.GROUP_PARTICIPANT_JOINED_GROUP_AND_PARENT_GROUP,
  WAMessageStubType.CAG_INVITE_AUTO_ADD,
  WAMessageStubType.CAG_INVITE_AUTO_JOINED,
  WAMessageStubType.SUB_GROUP_PARTICIPANT_ADD_RICH,
  WAMessageStubType.COMMUNITY_PARTICIPANT_ADD_RICH,
  WAMessageStubType.SUBGROUP_ADMIN_TRIGGERED_AUTO_ADD_RICH,
].filter((value) => Number.isInteger(value)));

function cacheSetBounded(map, key, value, maxSize = 5000) {
  map.set(key, value);
  if (map.size > maxSize) {
    const firstKey = map.keys().next().value;
    map.delete(firstKey);
  }
}

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

function isPathWithin(basePath, candidatePath) {
  const relative = path.relative(basePath, candidatePath);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
}

async function resolveAllowedAttachmentPath(rawPath) {
  if (typeof rawPath !== 'string' || !rawPath.trim()) {
    throw actionError('invalid_target', 'attachment path is required');
  }
  const candidate = path.resolve(rawPath.trim());
  if (!await fs.pathExists(candidate)) {
    throw actionError('not_found', `attachment not found: ${rawPath}`);
  }
  const [mediaDirRealPath, candidateRealPath] = await Promise.all([
    fs.realpath(config.mediaDir),
    fs.realpath(candidate),
  ]);
  if (!isPathWithin(mediaDirRealPath, candidateRealPath)) {
    throw actionError('invalid_target', `attachment path must be inside media dir: ${config.mediaDir}`);
  }
  const stat = await fs.stat(candidateRealPath);
  if (!stat.isFile()) {
    throw actionError('invalid_target', 'attachment path must point to a file');
  }
  return candidateRealPath;
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

function rememberSenderRef(chatId, senderId, participantJid = null) {
  if (!chatId || !senderId) return null;
  const canonicalSenderId = normalizeJid(senderId) || senderId;
  const canonicalParticipant = normalizeJid(participantJid) || participantJid || canonicalSenderId;
  const registry = ensureSenderRefRegistry(chatId);

  const existingRef = registry.senderToRef.get(canonicalSenderId);
  if (existingRef) {
    registry.senderToParticipant.set(canonicalSenderId, canonicalParticipant);
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

function normalizeJid(jid) {
  if (!jid || typeof jid !== 'string') return null;
  try {
    return jidNormalizedUser(jid);
  } catch {
    return jid;
  }
}

function rememberParticipantName(jid, name) {
  if (!jid || typeof jid !== 'string') return;
  if (!name || typeof name !== 'string') return;
  const cleaned = name.trim();
  if (!/[\p{L}\p{N}]/u.test(cleaned)) return;
  if (!cleaned) return;

  cacheSetBounded(participantNameCache, jid, cleaned);
  const normalized = normalizeJid(jid);
  if (normalized) cacheSetBounded(participantNameCache, normalized, cleaned);
}

function lookupParticipantName(jid) {
  if (!jid || typeof jid !== 'string') return null;
  const direct = participantNameCache.get(jid);
  if (direct) return direct;
  const normalized = normalizeJid(jid);
  if (!normalized) return null;
  return participantNameCache.get(normalized) || null;
}

function groupParticipantKey(chatId, participantJid) {
  const normalized = normalizeJid(participantJid) || participantJid;
  return `${chatId}::${normalized}`;
}

function participantDisplayName(participant) {
  if (!participant || typeof participant !== 'object') return null;
  const candidates = [
    participant.name,
    participant.notify,
    participant.pushName,
    participant.verifiedName,
    participant.vname,
  ];
  for (const candidate of candidates) {
    if (typeof candidate !== 'string') continue;
    const cleaned = candidate.trim();
    if (!/[\p{L}\p{N}]/u.test(cleaned)) continue;
    if (cleaned) return cleaned;
  }
  return null;
}

function hydrateGroupParticipantCaches(chatId, participants) {
  if (!chatId || !Array.isArray(participants)) return;
  for (const participant of participants) {
    const name = participantDisplayName(participant);
    if (!name) continue;
    const aliases = extractParticipantAliases(participant);
    for (const alias of aliases) {
      rememberParticipantName(alias, name);
      cacheSetBounded(groupParticipantNameCache, groupParticipantKey(chatId, alias), name);
    }
  }
}

function parseGroupDescription(rawDescription) {
  if (typeof rawDescription !== 'string' || !rawDescription.trim()) {
    return { description: null, promptOveride: null };
  }
  const blocks = [];
  const cleaned = rawDescription
    .replace(/<prompt_override>([\s\S]*?)<\/prompt_override>/gi, (_, block) => {
      const trimmed = typeof block === 'string' ? block.trim() : '';
      if (trimmed) blocks.push(trimmed);
      return '';
    })
    .replace(/\n{3,}/g, '\n\n')
    .trim();

  return {
    description: cleaned || null,
    promptOveride: blocks.length > 0 ? blocks.join('\n\n') : null,
  };
}

function pickGroupDescription(meta) {
  const candidates = [
    meta?.desc,
    meta?.description,
    meta?.descText,
  ];
  for (const candidate of candidates) {
    if (typeof candidate !== 'string') continue;
    const trimmed = candidate.trim();
    if (trimmed) return trimmed;
  }
  return null;
}

function participantRoleFlags(participant) {
  const adminRole = typeof participant?.admin === 'string' ? participant.admin.toLowerCase() : '';
  const isSuperAdmin = adminRole === 'superadmin';
  const isAdmin = isSuperAdmin || adminRole === 'admin';
  return { isAdmin, isSuperAdmin };
}

function buildParticipantRoleMap(meta) {
  const roleMap = {};
  const participants = Array.isArray(meta?.participants) ? meta.participants : [];
  for (const participant of participants) {
    const roleFlags = participantRoleFlags(participant);
    const aliases = extractParticipantAliases(participant);
    for (const alias of aliases) {
      if (!alias) continue;
      roleMap[alias] = roleFlags;
    }
  }
  return roleMap;
}

function currentBotAliases() {
  if (!sock?.user) return [];
  const aliases = extractParticipantAliases([
    sock.user.id,
    sock.user.jid,
    sock.user.lid,
    sock.user.phoneNumber,
  ]);
  if (aliases.length > 0) return aliases;
  const normalized = normalizeJid(sock.user.id);
  return normalized ? [normalized] : [];
}

function roleFlagsForJid(participantRoles, jid) {
  if (!participantRoles || typeof participantRoles !== 'object') {
    return { isAdmin: false, isSuperAdmin: false };
  }
  const normalized = normalizeJid(jid) || jid;
  if (!normalized) return { isAdmin: false, isSuperAdmin: false };
  const found = participantRoles[normalized];
  if (!found) return { isAdmin: false, isSuperAdmin: false };
  return {
    isAdmin: Boolean(found.isAdmin),
    isSuperAdmin: Boolean(found.isSuperAdmin),
  };
}

function normalizeGroupMetadata(meta, jid) {
  const name = meta?.subject || jid;
  const rawDescription = pickGroupDescription(meta);
  const { description, promptOveride } = parseGroupDescription(rawDescription || '');
  const participantRoles = buildParticipantRoleMap(meta);
  const botAliases = currentBotAliases();
  let botIsAdmin = false;
  let botIsSuperAdmin = false;
  for (const alias of botAliases) {
    const flags = roleFlagsForJid(participantRoles, alias);
    if (flags.isSuperAdmin) botIsSuperAdmin = true;
    if (flags.isAdmin || flags.isSuperAdmin) botIsAdmin = true;
  }
  return {
    name,
    description,
    promptOveride,
    botIsAdmin,
    botIsSuperAdmin,
    participantRoles,
  };
}

function getCachedGroupMetadata(jid) {
  const cached = groupMetadataCache.get(jid);
  if (!cached) return null;
  if (Date.now() - cached.fetchedAt > GROUP_METADATA_TTL_MS) {
    groupMetadataCache.delete(jid);
    return null;
  }
  return cached.value;
}

function rememberGroupMetadata(jid, value) {
  cacheSetBounded(groupMetadataCache, jid, {
    fetchedAt: Date.now(),
    value,
  }, 2000);
}

function invalidateGroupMetadata(jid) {
  if (!jid) return;
  groupMetadataCache.delete(jid);
}

async function getGroupContext(jid, { forceRefresh = false } = {}) {
  if (!jid) return { name: jid, description: null, promptOveride: null, botIsAdmin: false, botIsSuperAdmin: false, participantRoles: {} };
  if (!sock) return { name: jid, description: null, promptOveride: null, botIsAdmin: false, botIsSuperAdmin: false, participantRoles: {} };

  if (!forceRefresh) {
    const cached = getCachedGroupMetadata(jid);
    if (cached) return cached;
  }

  try {
    const meta = await sock.groupMetadata(jid);
    hydrateGroupParticipantCaches(jid, meta?.participants);
    const normalized = normalizeGroupMetadata(meta, jid);
    rememberGroupMetadata(jid, normalized);
    return normalized;
  } catch (err) {
    logger.warn({ err, jid }, 'failed to fetch group metadata');
    const cached = getCachedGroupMetadata(jid);
    if (cached) return cached;
    return { name: jid, description: null, promptOveride: null, botIsAdmin: false, botIsSuperAdmin: false, participantRoles: {} };
  }
}

async function getGroupParticipantName(chatId, participantJid) {
  if (!sock || !chatId || !participantJid) return null;
  const key = groupParticipantKey(chatId, participantJid);
  const cached = groupParticipantNameCache.get(key);
  if (cached) return cached;

  const fallback = lookupParticipantName(participantJid);
  if (fallback) {
    cacheSetBounded(groupParticipantNameCache, key, fallback);
    return fallback;
  }

  const hadCachedMetadata = Boolean(getCachedGroupMetadata(chatId));
  await getGroupContext(chatId);
  let resolved = lookupParticipantName(participantJid);

  if (!resolved && hadCachedMetadata) {
    await getGroupContext(chatId, { forceRefresh: true });
    resolved = lookupParticipantName(participantJid);
  }

  if (resolved) {
    cacheSetBounded(groupParticipantNameCache, key, resolved);
  }
  return resolved || null;
}

function inferExtension(mime) {
  if (!mime) return 'bin';
  if (mime.includes('jpeg')) return 'jpg';
  if (mime.includes('png')) return 'png';
  if (mime.includes('gif')) return 'gif';
  if (mime.includes('webp')) return 'webp';
  if (mime.includes('mp4')) return 'mp4';
  if (mime.includes('mp3')) return 'mp3';
  if (mime.includes('ogg')) return 'ogg';
  if (mime.includes('pdf')) return 'pdf';
  if (mime.includes('zip')) return 'zip';
  return mime.split('/').pop();
}

function mapMediaKind(contentType) {
  if (contentType === 'imageMessage') return 'image';
  if (contentType === 'videoMessage') return 'video';
  if (contentType === 'audioMessage') return 'audio';
  if (contentType === 'documentMessage') return 'document';
  if (contentType === 'stickerMessage') return 'sticker';
  return 'unknown';
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

function unwrapMessage(message) {
  if (!message) return { contentType: null, message: null };
  const normalized = normalizeMessageContent(message);
  const contentType = normalized ? getContentType(normalized) : null;
  return { contentType, message: normalized };
}

function extractContextInfo(message) {
  if (!message) return undefined;
  const contentType = getContentType(message);
  const candidate = contentType ? message[contentType] : undefined;
  if (candidate?.contextInfo) return candidate.contextInfo;
  // fall back: scan nested entries
  for (const value of Object.values(message)) {
    if (value && typeof value === 'object' && 'contextInfo' in value) {
      return value.contextInfo;
    }
  }
  return undefined;
}

function extractMentionedJids(message) {
  const ctx = extractContextInfo(message);
  const mentions = ctx?.mentionedJid;
  if (!mentions || mentions.length === 0) return null;
  return Array.from(new Set(mentions));
}

function parseVcardPhones(vcard) {
  if (!vcard) return [];
  const lines = vcard.split(/\r?\n/);
  const phones = [];
  for (const line of lines) {
    const match = line.match(/^TEL[^:]*:(.+)$/i);
    if (match?.[1]) phones.push(match[1].trim());
  }
  return phones;
}

function extractContactPlaceholder(message) {
  if (message?.contactMessage) {
    const { displayName, vcard } = message.contactMessage;
    const phones = parseVcardPhones(vcard);
    const label = [displayName, phones[0]].filter(Boolean).join(', ');
    return label ? `<contact: ${label}>` : '<contact>';
  }
  const contacts = message?.contactsArrayMessage?.contacts;
  if (contacts && contacts.length > 0) {
    const first = contacts[0];
    const name = first?.displayName;
    const phones = parseVcardPhones(first?.vcard || '');
    const label = [name, phones[0]].filter(Boolean).join(', ');
    const suffix = contacts.length > 1 ? ` +${contacts.length - 1} more` : '';
    return label ? `<contacts: ${label}${suffix}>` : `<contacts: ${contacts.length}>`;
  }
  return null;
}

function extractLocationData(message) {
  if (!message) return null;
  const live = message.liveLocationMessage;
  if (live?.degreesLatitude != null && live?.degreesLongitude != null) {
    return {
      latitude: Number(live.degreesLatitude),
      longitude: Number(live.degreesLongitude),
      accuracy: live.accuracyInMeters,
      caption: live.caption,
      isLive: true,
    };
  }
  const location = message.locationMessage;
  if (location?.degreesLatitude != null && location?.degreesLongitude != null) {
    return {
      latitude: Number(location.degreesLatitude),
      longitude: Number(location.degreesLongitude),
      accuracy: location.accuracyInMeters,
      name: location.name,
      address: location.address,
      caption: location.comment,
      isLive: Boolean(location.isLive),
    };
  }
  return null;
}

function formatLocationText(loc) {
  const parts = [];
  if (loc.name) parts.push(loc.name);
  if (loc.address && loc.address !== loc.name) parts.push(loc.address);
  const coords = Number.isFinite(loc.latitude) && Number.isFinite(loc.longitude)
    ? `${loc.latitude.toFixed(5)}, ${loc.longitude.toFixed(5)}`
    : null;
  if (coords) parts.push(coords);
  if (loc.caption) parts.push(loc.caption);
  return parts.length ? `📍 ${parts.join(' | ')}` : '📍 Location';
}

function extractInteractiveText(message) {
  if (!message) return null;
  const btn = message.buttonsResponseMessage;
  if (btn) return btn.selectedDisplayText || btn.selectedButtonId || btn.selectedId || null;

  const tmpl = message.templateButtonReplyMessage;
  if (tmpl) return tmpl.selectedDisplayText || tmpl.selectedId || String(tmpl.selectedIndex ?? '');

  const list = message.listResponseMessage;
  if (list) {
    return (
      list.title ||
      list.description ||
      list.singleSelectReply?.title ||
      list.singleSelectReply?.description ||
      list.singleSelectReply?.selectedRowId ||
      null
    );
  }

  const interactive = message.interactiveResponseMessage;
  if (interactive?.nativeFlowResponseMessage?.paramsJson) {
    try {
      const parsed = JSON.parse(interactive.nativeFlowResponseMessage.paramsJson);
      if (typeof parsed === 'string') return parsed;
      if (parsed?.id) return parsed.id;
      if (parsed?.selection?.title) return parsed.selection.title;
      if (parsed?.selection?.id) return parsed.selection.id;
      if (parsed?.name) return parsed.name;
    } catch (err) {
      logger.debug({ err }, 'failed to parse nativeFlowResponse paramsJson');
    }
    return interactive.nativeFlowResponseMessage.paramsJson;
  }
  if (interactive?.body) return interactive.body;

  return null;
}

function extractMediaPlaceholder(message) {
  if (!message) return null;
  if (message.imageMessage) return '<media:image>';
  if (message.videoMessage) return '<media:video>';
  if (message.audioMessage) return '<media:audio>';
  if (message.documentMessage) return '<media:document>';
  if (message.stickerMessage) return '<media:sticker>';
  return null;
}

function extractText(message) {
  if (!message) return null;

  const text = message.conversation?.trim();
  if (text) return text;

  const extended = message.extendedTextMessage?.text?.trim();
  if (extended) return extended;

  const interactive = extractInteractiveText(message);
  if (interactive) return interactive;

  const caption =
    message.imageMessage?.caption || message.videoMessage?.caption || message.documentMessage?.caption;
  if (caption) return caption;

  const reaction = message.reactionMessage?.text;
  if (reaction) return `react:${reaction}`;

  const contact = extractContactPlaceholder(message);
  if (contact) return contact;

  const mediaPlaceholder = extractMediaPlaceholder(message);
  if (mediaPlaceholder) return mediaPlaceholder;

  return null;
}

async function saveMedia(contentType, content, messageId) {
  const kind = mapMediaKind(contentType);
  if (kind === 'unknown') return null;
  const mime = content.mimetype || 'application/octet-stream';
  const ext = inferExtension(mime);
  const filename = `${messageId}_${kind}.${ext}`;
  const filepath = path.join(config.mediaDir, filename);

  const stream = await downloadContentFromMessage(content, kind);
  const size = await streamToFile(stream, filepath);

  return {
    kind,
    mime,
    fileName: filename,
    size,
    path: filepath,
    isAnimated: Boolean(content.isAnimated),
  };
}

async function extractQuoted(messageOrContent, chatId) {
  const ctx = extractContextInfo(messageOrContent);
  if (!ctx || !ctx.quotedMessage) return null;
  const { contentType: qType, message: qMsg } = unwrapMessage(ctx.quotedMessage);
  if (!qMsg) return null;
  const location = extractLocationData(qMsg);
  const locationText = location ? formatLocationText(location) : null;
  const qText = extractText(qMsg);
  const text = [qText, locationText].filter(Boolean).join('\n') || null;
  let senderId = ctx.participant ? normalizeJid(ctx.participant) : null;
  let senderName = null;
  const quotedMsg = ctx.stanzaId ? messageCache.get(ctx.stanzaId) : null;

  if (quotedMsg) {
    const quotedFromId = quotedMsg.key?.participant || quotedMsg.key?.remoteJid;
    if (!senderId && quotedFromId) {
      senderId = normalizeJid(quotedFromId);
    }
    const quotedPushName = quotedMsg.pushName;
    if (typeof quotedPushName === 'string' && quotedPushName.trim()) {
      senderName = quotedPushName.trim();
      if (senderId) rememberParticipantName(senderId, senderName);
      if (quotedFromId) rememberParticipantName(quotedFromId, senderName);
    }
  }

  if (!senderName && senderId) senderName = lookupParticipantName(senderId);
  if (!senderName && ctx.participant) senderName = lookupParticipantName(ctx.participant);
  if (!senderName && chatId?.endsWith('@g.us') && senderId) {
    senderName = await getGroupParticipantName(chatId, senderId);
  }
  const contextMsgId = ctx.stanzaId ? findContextMsgIdByMessageId(chatId, ctx.stanzaId) : null;

  return {
    messageId: ctx.stanzaId,
    contextMsgId,
    senderId,
    senderName: senderName || senderId,
    text,
    type: qType,
    location,
  };
}

function fallbackParticipantLabel(jid) {
  if (!jid || typeof jid !== 'string') return 'unknown';
  const local = jid.split('@')[0] || jid;
  if (!local) return 'unknown';
  const digits = local.replace(/\D/g, '');
  if (digits.length >= 5) return digits;
  return local;
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

function makeEventMessageId(prefix) {
  const stamp = Date.now();
  const rand = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${stamp}_${rand}`;
}

function toJidCandidate(value) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.includes('@')) return trimmed;

  const digits = trimmed.replace(/\D/g, '');
  if (digits.length >= 5) return `${digits}@s.whatsapp.net`;
  return null;
}

function choosePreferredParticipantJid(jids) {
  if (!Array.isArray(jids) || jids.length === 0) return null;
  const unique = Array.from(new Set(jids.filter((jid) => typeof jid === 'string' && jid.trim())));
  if (unique.length === 0) return null;
  const pn = unique.find((jid) => jid.endsWith('@s.whatsapp.net') || jid.endsWith('@c.us'));
  return pn || unique[0];
}

function extractParticipantAliases(value) {
  if (!value) return [];
  if (Array.isArray(value)) {
    const normalized = [];
    for (const item of value) {
      const aliases = extractParticipantAliases(item);
      normalized.push(...aliases);
    }
    return Array.from(new Set(normalized));
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return [];
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        return extractParticipantAliases(JSON.parse(trimmed));
      } catch {
        return [];
      }
    }
    const parsed = toJidCandidate(trimmed);
    if (!parsed) return [];
    const cleaned = normalizeJid(parsed) || parsed;
    return [cleaned];
  }

  if (typeof value !== 'object') return [];
  const candidates = [
    value.phoneNumber,
    value.pn,
    value.id,
    value.jid,
    value.lid,
    value.participant,
  ];

  const normalized = [];
  for (const candidate of candidates) {
    const parsed = toJidCandidate(candidate);
    if (!parsed) continue;
    normalized.push(normalizeJid(parsed) || parsed);
  }
  return Array.from(new Set(normalized));
}

function extractParticipantJids(value) {
  if (!value) return [];
  if (Array.isArray(value)) {
    const normalized = [];
    for (const item of value) {
      const aliases = extractParticipantAliases(item);
      const preferred = choosePreferredParticipantJid(aliases);
      if (preferred) normalized.push(preferred);
    }
    return Array.from(new Set(normalized));
  }

  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return [];
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      try {
        return extractParticipantJids(JSON.parse(trimmed));
      } catch {
        return [];
      }
    }
    const aliases = extractParticipantAliases(trimmed);
    const preferred = choosePreferredParticipantJid(aliases);
    return preferred ? [preferred] : [];
  }

  const aliases = extractParticipantAliases(value);
  const preferred = choosePreferredParticipantJid(aliases);
  return preferred ? [preferred] : [];
}

function compactParticipantJids(participants) {
  if (!Array.isArray(participants)) return [];
  const normalized = [];
  for (const participant of participants) {
    const candidates = extractParticipantJids(participant);
    for (const jid of candidates) {
      const cleaned = normalizeJid(jid) || jid;
      normalized.push(cleaned);
    }
  }
  return Array.from(new Set(normalized));
}

function normalizeGroupJoinAction(action) {
  const normalized = typeof action === 'string' ? action.toLowerCase() : '';
  if (!normalized) return 'join';
  if (normalized === 'add' || normalized.includes('_add')) return 'add';
  if (normalized === 'invite' || normalized.includes('invite')) return 'invite';
  if (normalized === 'approve' || normalized === 'accept' || normalized.includes('approve') || normalized.includes('accept')) {
    return 'approve';
  }
  if (normalized === 'join' || normalized.includes('join')) return 'join';
  return normalized;
}

function dedupeGroupJoinEvent(chatId, participants, action, timestampMs) {
  const ts = Number(timestampMs) || Date.now();
  const normalizedAction = normalizeGroupJoinAction(action);
  const normalizedParticipants = compactParticipantJids(participants).sort();
  const key = `${chatId}::${normalizedAction}::${normalizedParticipants.join(',')}`;
  const lastSeen = groupJoinDedupCache.get(key);
  if (lastSeen && ts - lastSeen < GROUP_JOIN_DEDUP_TTL_MS) {
    return false;
  }
  cacheSetBounded(groupJoinDedupCache, key, ts, 2000);
  return true;
}

function stubActionName(stubType) {
  if (typeof stubType !== 'number') return 'join';
  const enumName = WAMessageStubType[stubType];
  if (typeof enumName !== 'string' || !enumName) return 'join';
  return enumName.toLowerCase();
}

function parseGroupJoinStub(msg) {
  const chatId = msg?.key?.remoteJid;
  if (!chatId || !chatId.endsWith('@g.us')) return null;
  const stubType = msg?.messageStubType;
  if (!GROUP_JOIN_STUB_TYPES.has(stubType)) return null;

  const rawParams = Array.isArray(msg?.messageStubParameters)
    ? msg.messageStubParameters
    : [];
  const parsedFromParams = compactParticipantJids(rawParams);
  const participants = parsedFromParams.length > 0
    ? parsedFromParams
    : compactParticipantJids([msg?.participant]);

  if (participants.length === 0) return null;

  const actorId = compactParticipantJids([
    msg?.key?.participantAlt,
    msg?.participantPn,
    msg?.key?.participant,
    msg?.participant,
  ])[0] || null;
  const timestampMs = Number(msg?.messageTimestamp) > 0
    ? Number(msg.messageTimestamp) * 1000
    : Date.now();
  return {
    chatId,
    action: normalizeGroupJoinAction(stubActionName(stubType)),
    participants,
    actorId,
    timestampMs,
    messageId: msg?.key?.id || null,
    messageKey: msg?.key?.id ? { ...msg.key } : null,
    source: 'messages.upsert.stub',
  };
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
  const uniqueLabels = Array.from(new Set(labels.filter(Boolean)));
  const normalizedActorId = normalizeJid(actorId) || null;
  const actorName = normalizedActorId
    ? await resolveParticipantLabel(chatId, normalizedActorId)
    : null;
  const actorSenderId = normalizedActorId || 'group-system@wazzap.local';
  const senderRef = rememberSenderRef(chatId, actorSenderId, actorSenderId) || 'unknown';
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
      senderRef,
      senderIsAdmin: actorRole.isAdmin || actorRole.isSuperAdmin,
      fromMe: false,
      timestampMs: normalizedTimestampMs,
    });
  }

  const joinedText = uniqueLabels.length === 1
    ? `${uniqueLabels[0]} joined the group.`
    : `New members joined the group: ${uniqueLabels.join(', ')}.`;
  const byText = actorName ? ` Added by ${actorName}.` : '';
  const text = `Group update: ${joinedText}${byText}`;

  const payload = {
    messageId: resolvedMessageId,
    instanceId: config.instanceId,
    chatId,
    chatName: group.name || chatId,
    chatType: 'group',
    senderId: actorSenderId,
    senderRef,
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
  const fromId = msg.key.participant || (fromMe ? selfJid : msg.key.remoteJid);
  const senderId = normalizeJid(fromId) || fromId || normalizeJid(msg.key.remoteJid) || msg.key.remoteJid;
  const senderDisplay = msg.pushName || lookupParticipantName(senderId) || senderId;
  rememberParticipantName(fromId, msg.pushName || '');
  rememberParticipantName(senderId, senderDisplay);

  const groupStartMs = Date.now();
  const group = isGroup ? await getGroupContext(chatId) : null;
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
  const quoted = await extractQuoted(innerMessage, chatId);
  perf.quotedMs = Date.now() - quotedStartMs;
  const mentionedJids = extractMentionedJids(innerMessage);

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
      const mediaInfo = await saveMedia(contentType, content, msg.key.id);
      if (mediaInfo) attachments.push(mediaInfo);
    } catch (err) {
      logger.error({ err }, 'failed saving media');
    } finally {
      perf.mediaMs = Date.now() - mediaStartMs;
    }
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
    isGroup,
    botIsAdmin: Boolean(group?.botIsAdmin),
    botIsSuperAdmin: Boolean(group?.botIsSuperAdmin),
    fromMe,
    contextOnly: fromMe,
    triggerLlm1: false,
    timestampMs: Number(msg.messageTimestamp) * 1000,
    messageType: contentType,
    text,
    quoted,
    attachments,
    mentionedJids,
    location,
    groupDescription: group?.description || null,
    groupPromptOveride: group?.promptOveride || null,
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
    return {
      contextMsgId: normalizedContextMsgId,
      messageId: indexed.id,
    };
  } catch (err) {
    throw actionError('send_failed', err?.message || 'failed to delete message');
  }
}

function normalizeKickTargets(rawTargets) {
  if (!Array.isArray(rawTargets)) return [];
  const normalized = [];
  for (const target of rawTargets) {
    const senderRef = typeof target?.senderRef === 'string'
      ? target.senderRef.trim().toLowerCase()
      : '';
    const anchorContextMsgId = normalizeContextMsgId(target?.anchorContextMsgId);
    normalized.push({
      senderRef,
      anchorContextMsgId,
    });
  }
  return normalized;
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
    defaultQueryTimeoutMs: 60_000,
  });

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

    if (isNotify) {
      for (const msg of messages) {
        const chatId = msg?.key?.remoteJid;
        const messageId = msg?.key?.id;
        if (!chatId || !messageId || chatId === 'status@broadcast') continue;
        if (GROUP_JOIN_STUB_TYPES.has(msg?.messageStubType) || !msg?.message) continue;
        const contextMsgId = ensureContextMsgId(chatId, messageId);
        precomputedContextByMessage.set(messageIdIndexKey(chatId, messageId), contextMsgId);
      }
    }

    await runWithConcurrency(messages, config.upsertConcurrency, async (msg) => {
      try {
        if (!isNotify) {
          const stubEvent = parseGroupJoinStub(msg);
          if (stubEvent) {
            await emitGroupJoinContextEvent(stubEvent);
          }
          return;
        }

        const chatId = msg?.key?.remoteJid;
        const messageId = msg?.key?.id;
        const precomputedContextMsgId = (chatId && messageId)
          ? precomputedContextByMessage.get(messageIdIndexKey(chatId, messageId))
          : null;
        await handleIncomingMessage(msg, { precomputedContextMsgId });
      } catch (err) {
        logger.error({ err }, 'failed handling message');
      }
    });

    const batchTotalMs = Date.now() - batchStartMs;
    if (config.perfLogEnabled && messages.length > 1 && batchTotalMs >= config.perfLogThresholdMs) {
      logger.info({
        type,
        messageCount: messages.length,
        upsertConcurrency: config.upsertConcurrency,
        batchTotalMs,
      }, 'slow messages.upsert batch');
    }
  });

  return sock;
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
  const group = isGroup ? await getGroupContext(chatId) : null;
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
    const filePath = await resolveAllowedAttachmentPath(att?.path);
    const content = {};
    if (kind === 'image') content.image = { url: filePath };
    else if (kind === 'video') content.video = { url: filePath };
    else if (kind === 'audio') content.audio = { url: filePath, ptt: false };
    else if (kind === 'sticker') content.sticker = { url: filePath };
    else content.document = { url: filePath, fileName: att.fileName || path.basename(filePath) };

    if (att.caption) content.caption = att.caption;

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
    const sentMsg = await sock.sendMessage(chatId, { text: normalizedText }, quoted ? { quoted } : {});
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

export {
  startWhatsApp,
  sendOutgoing,
  deleteMessageByContextId,
  kickMembers,
};
