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
} from 'baileys';
import logger from './logger.js';
import config from './config.js';
import { streamToBuffer } from './utils.js';
import wsClient from './wsClient.js';

let sock;
const messageCache = new Map(); // simple in-memory store for quoting outbound
const MAX_CACHE = 200;
const groupNameCache = new Map();
const participantNameCache = new Map();
const groupParticipantNameCache = new Map();

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
    if (cleaned) return cleaned;
  }
  return null;
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
      const pid = participant?.id;
      if (!pid) continue;
      const name = participantDisplayName(participant);
      if (!name) continue;
      rememberParticipantName(pid, name);
      cacheSetBounded(groupParticipantNameCache, groupParticipantKey(chatId, pid), name);
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

async function getGroupName(jid) {
  if (!sock) return jid;
  if (groupNameCache.has(jid)) return groupNameCache.get(jid);
  try {
    const meta = await sock.groupMetadata(jid);
    const name = meta?.subject || jid;
    groupNameCache.set(jid, name);
    return name;
  } catch (err) {
    logger.warn({ err, jid }, 'failed to fetch group metadata');
    return jid;
  }
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

async function handleIncomingMessage(msg) {
  if (!msg.message) return;
  if (!sock) return;
  const remoteJid = msg.key.remoteJid;
  if (!remoteJid || remoteJid === 'status@broadcast') return;

  const fromId = msg.key.participant || msg.key.remoteJid;
  const senderId = jidNormalizedUser(fromId);
  const senderName = msg.pushName || senderId;
  rememberParticipantName(fromId, msg.pushName || '');
  rememberParticipantName(senderId, senderName);
  const chatId = remoteJid;
  const isGroup = chatId.endsWith('@g.us');
  const chatName = isGroup ? await getGroupName(chatId) : (msg.pushName || chatId);
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

  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;
    for (const msg of messages) {
      try {
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
