import path from 'path';
import fs from 'fs-extra';
import makeWASocket, {
  fetchLatestBaileysVersion,
  useMultiFileAuthState,
  DisconnectReason,
  getContentType,
  downloadContentFromMessage,
  jidNormalizedUser,
} from 'baileys';
import logger from './logger.js';
import config from './config.js';
import { streamToBuffer } from './utils.js';
import wsClient from './wsClient.js';

let sock;
const messageCache = new Map(); // simple in-memory store for quoting outbound
const MAX_CACHE = 200;
const groupNameCache = new Map();

function rememberMessage(msg) {
  if (!msg?.key?.id) return;
  messageCache.set(msg.key.id, msg);
  if (messageCache.size > MAX_CACHE) {
    const firstKey = messageCache.keys().next().value;
    messageCache.delete(firstKey);
  }
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

function extractTextFromContent(contentType, message) {
  if (!message) return null;
  const content = message[contentType];
  if (!content) return null;
  if (content.text) return content.text;
  if (content.caption) return content.caption;
  if (contentType === 'conversation') return content;
  if (contentType === 'extendedTextMessage') return content.text || null;
  return null;
}

function unwrapMessage(message) {
  if (!message) return { contentType: null, message: null };
  let current = message;
  let contentType = getContentType(current);
  while (contentType === 'ephemeralMessage' || contentType === 'viewOnceMessageV2' || contentType === 'viewOnceMessage') {
    current = current[contentType]?.message;
    contentType = current ? getContentType(current) : null;
  }
  return { contentType, message: current };
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

function extractQuoted(content) {
  const ctx = content?.contextInfo;
  if (!ctx || !ctx.quotedMessage) return null;
  const { contentType: qType, message: qMsg } = unwrapMessage(ctx.quotedMessage);
  const qText = extractTextFromContent(qType, qMsg);
  return {
    messageId: ctx.stanzaId,
    senderId: ctx.participant,
    text: qText,
    type: qType,
  };
}

async function handleIncomingMessage(msg) {
  if (!msg.message) return;
  if (!sock) return;
  const remoteJid = msg.key.remoteJid;
  if (!remoteJid || remoteJid === 'status@broadcast') return;

  const fromId = msg.key.participant || msg.key.remoteJid;
  const senderId = jidNormalizedUser(fromId);
  const chatId = remoteJid;
  const isGroup = chatId.endsWith('@g.us');
  const chatName = isGroup ? await getGroupName(chatId) : (msg.pushName || chatId);
  const { contentType, message: innerMessage } = unwrapMessage(msg.message);
  if (!contentType || !innerMessage) return;
  const content = innerMessage[contentType];
  const text = extractTextFromContent(contentType, innerMessage);
  const quoted = extractQuoted(content);

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
    senderName: msg.pushName || senderId,
    isGroup,
    timestampMs: Number(msg.messageTimestamp) * 1000,
    messageType: contentType,
    text: text || null,
    quoted,
    attachments,
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
    printQRInTerminal: true,
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
