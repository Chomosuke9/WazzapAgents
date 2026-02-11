import path from 'path';
import fs from 'fs-extra';
import { spawn } from 'child_process';
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
import { streamToBuffer } from './utils.js';
import wsClient from './wsClient.js';

let sock;
const messageCache = new Map(); // simple in-memory store for quoting outbound
const MAX_CACHE = 200;
const GROUP_METADATA_TTL_MS = 60_000;
const GROUP_JOIN_DEDUP_TTL_MS = 15_000;
const groupMetadataCache = new Map();
const participantNameCache = new Map();
const groupParticipantNameCache = new Map();
const groupJoinDedupCache = new Map();
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

function rememberMessage(msg) {
  if (!msg?.key?.id) return;
  messageCache.set(msg.key.id, msg);
  if (messageCache.size > MAX_CACHE) {
    const firstKey = messageCache.keys().next().value;
    messageCache.delete(firstKey);
  }
}

function cacheSetBounded(map, key, value, maxSize = 5000) {
  map.set(key, value);
  if (map.size > maxSize) {
    const firstKey = map.keys().next().value;
    map.delete(firstKey);
  }
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

function normalizeGroupMetadata(meta, jid) {
  const name = meta?.subject || jid;
  const rawDescription = pickGroupDescription(meta);
  const { description, promptOveride } = parseGroupDescription(rawDescription || '');
  return {
    name,
    description,
    promptOveride,
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
  if (!jid) return { name: jid, description: null, promptOveride: null };
  if (!sock) return { name: jid, description: null, promptOveride: null };

  if (!forceRefresh) {
    const cached = getCachedGroupMetadata(jid);
    if (cached) return cached;
  }

  try {
    const meta = await sock.groupMetadata(jid);
    const normalized = normalizeGroupMetadata(meta, jid);
    rememberGroupMetadata(jid, normalized);
    return normalized;
  } catch (err) {
    logger.warn({ err, jid }, 'failed to fetch group metadata');
    const cached = getCachedGroupMetadata(jid);
    if (cached) return cached;
    return { name: jid, description: null, promptOveride: null };
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

  try {
    const meta = await sock.groupMetadata(chatId);
    const participants = meta?.participants || [];
    for (const participant of participants) {
      const name = participantDisplayName(participant);
      if (!name) continue;
      const aliases = extractParticipantAliases(participant);
      for (const alias of aliases) {
        rememberParticipantName(alias, name);
        cacheSetBounded(groupParticipantNameCache, groupParticipantKey(chatId, alias), name);
      }
    }
  } catch (err) {
    logger.debug({ err, chatId, participantJid }, 'failed resolving group participant name');
  }

  const resolved = lookupParticipantName(participantJid);
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
  const buffer = await streamToBuffer(stream);
  await fs.writeFile(filepath, buffer);

  return {
    kind,
    mime,
    fileName: filename,
    size: buffer.length,
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

  return {
    messageId: ctx.stanzaId,
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
    messageId: msg?.key?.id ? `${msg.key.id}_join_ctx` : null,
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
  source = 'group-participants.update',
}) {
  const normalizedParticipants = compactParticipantJids(participants);
  if (!chatId || !chatId.endsWith('@g.us') || normalizedParticipants.length === 0) return;
  if (!dedupeGroupJoinEvent(chatId, normalizedParticipants, action, timestampMs)) return;

  const group = await getGroupContext(chatId, { forceRefresh: true });
  const labels = [];
  for (const participantJid of normalizedParticipants) {
    const label = await resolveParticipantLabel(chatId, participantJid);
    labels.push(label);
  }
  const uniqueLabels = Array.from(new Set(labels.filter(Boolean)));
  const normalizedActorId = normalizeJid(actorId) || null;
  const actorName = normalizedActorId
    ? await resolveParticipantLabel(chatId, normalizedActorId)
    : null;

  const joinedText = uniqueLabels.length === 1
    ? `${uniqueLabels[0]} joined the group.`
    : `New members joined the group: ${uniqueLabels.join(', ')}.`;
  const byText = actorName ? ` Added by ${actorName}.` : '';
  const text = `Group update: ${joinedText}${byText}`;

  const payload = {
    messageId: messageId || makeEventMessageId('group_join'),
    instanceId: config.instanceId,
    chatId,
    chatName: group.name || chatId,
    senderId: normalizedActorId || 'group-system@wazzap.local',
    senderName: actorName || 'Group System',
    isGroup: true,
    timestampMs: Number(timestampMs) || Date.now(),
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

  wsClient.send({ type: 'incoming_message', payload });
}

async function handleGroupParticipantsUpdate(update) {
  if (!sock) return;
  const chatId = update?.id;
  if (!chatId || !chatId.endsWith('@g.us')) return;

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

async function handleIncomingMessage(msg) {
  if (!sock) return;
  const stubEvent = parseGroupJoinStub(msg);
  if (stubEvent) {
    await emitGroupJoinContextEvent(stubEvent);
    return;
  }

  if (!msg.message) return;
  const remoteJid = msg.key.remoteJid;
  if (!remoteJid || remoteJid === 'status@broadcast') return;

  const fromId = msg.key.participant || msg.key.remoteJid;
  const senderId = jidNormalizedUser(fromId);
  const senderName = msg.pushName || senderId;
  rememberParticipantName(fromId, msg.pushName || '');
  rememberParticipantName(senderId, senderName);
  const chatId = remoteJid;
  const isGroup = chatId.endsWith('@g.us');
  const group = isGroup ? await getGroupContext(chatId) : null;
  const chatName = isGroup ? (group?.name || chatId) : (msg.pushName || chatId);
  const { contentType, message: innerMessage } = unwrapMessage(msg.message);
  if (!contentType || !innerMessage) return;
  const content = innerMessage[contentType];
  const location = extractLocationData(innerMessage);
  const locationText = location ? formatLocationText(location) : null;
  const baseText = extractText(innerMessage);
  const text = [baseText, locationText].filter(Boolean).join('\n') || null;
  const quoted = await extractQuoted(innerMessage, chatId);
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
    try {
      const mediaInfo = await saveMedia(contentType, content, msg.key.id);
      if (mediaInfo) attachments.push(mediaInfo);
    } catch (err) {
      logger.error({ err }, 'failed saving media');
    }
  }

  const payload = {
    messageId: msg.key.id,
    instanceId: config.instanceId,
    chatId,
    chatName,
    senderId,
    senderName,
    isGroup,
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
  rememberMessage(msg);
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
    const isNotify = type === 'notify';
    for (const msg of messages) {
      try {
        if (!isNotify) {
          const stubEvent = parseGroupJoinStub(msg);
          if (stubEvent) {
            await emitGroupJoinContextEvent(stubEvent);
          }
          continue;
        }
        await handleIncomingMessage(msg);
      } catch (err) {
        logger.error({ err }, 'failed handling message');
      }
    }
  });

  return sock;
}

async function sendOutgoing({ chatId, text, attachments = [], replyTo }) {
  if (!sock) throw new Error('WhatsApp socket not ready');

  const quoted = replyTo ? messageCache.get(replyTo) : null;

  // send attachments first (with caption if provided)
  for (const att of attachments) {
    const kind = att.kind || att.type || 'document';
    const filePath = att.path;
    if (!filePath) {
      logger.warn({ att }, 'attachment missing path, skipped');
      continue;
    }
    const content = {};
    if (kind === 'image') content.image = { url: filePath };
    else if (kind === 'video') content.video = { url: filePath };
    else if (kind === 'audio') content.audio = { url: filePath, ptt: false };
    else if (kind === 'sticker') content.sticker = { url: filePath };
    else content.document = { url: filePath, fileName: att.fileName || path.basename(filePath) };

    if (att.caption) content.caption = att.caption;

    await sock.sendMessage(chatId, content, quoted ? { quoted } : {});
  }

  if (text) {
    await sock.sendMessage(chatId, { text }, quoted ? { quoted } : {});
  }
}

export {
  startWhatsApp,
  sendOutgoing,
};
