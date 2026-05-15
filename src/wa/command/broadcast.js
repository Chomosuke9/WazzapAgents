import logger from '../../logger.js';
import { isOwnerJid } from '../../participants.js';
import { messageCache } from '../../caches.js';
import { getSock } from '../connection.js';
import { sendRichMessage } from '../interactive/index.js';

async function reconstructAndSend(sock, targetJid, cachedMsg) {
  const msg = cachedMsg.message;
  if (!msg) {
    logger.warn({ targetJid }, 'reconstructAndSend: cachedMsg.message is empty');
    return { ok: false, reason: 'error' };
  }

  try {
    if (msg.conversation || msg.extendedTextMessage) {
      const text = msg.conversation || msg.extendedTextMessage?.text;
      const mentions = msg.extendedTextMessage?.contextInfo?.mentionedJid || [];
      const content = { text };
      if (mentions.length > 0) content.mentions = mentions;
      await sock.sendMessage(targetJid, content);
      return { ok: true };
    }

    // Media keys from the original message are reused as-is. If the WhatsApp CDN validates
    // keys against the source chat session, recipients in different groups may see silent
    // download failures. Validate with a real multi-group test before relying on media broadcast.

    if (msg.imageMessage) {
      const img = msg.imageMessage;
      const content = {
        image: {
          url: img.url,
          directPath: img.directPath,
          mediaKey: img.mediaKey,
          fileEncSha256: img.fileEncSha256,
          fileSha256: img.fileSha256,
          fileLength: img.fileLength
        },
        mimetype: img.mimetype
      };
      if (img.caption) content.caption = img.caption;
      const mentions = img.contextInfo?.mentionedJid || [];
      if (mentions.length > 0) content.mentions = mentions;
      await sock.sendMessage(targetJid, content);
      return { ok: true };
    }

    if (msg.videoMessage) {
      const vid = msg.videoMessage;
      const content = {
        video: {
          url: vid.url,
          directPath: vid.directPath,
          mediaKey: vid.mediaKey,
          fileEncSha256: vid.fileEncSha256,
          fileSha256: vid.fileSha256,
          fileLength: vid.fileLength
        },
        mimetype: vid.mimetype
      };
      if (vid.caption) content.caption = vid.caption;
      const mentions = vid.contextInfo?.mentionedJid || [];
      if (mentions.length > 0) content.mentions = mentions;
      await sock.sendMessage(targetJid, content);
      return { ok: true };
    }

    if (msg.audioMessage) {
      const audio = msg.audioMessage;
      const content = {
        audio: {
          url: audio.url,
          directPath: audio.directPath,
          mediaKey: audio.mediaKey,
          fileEncSha256: audio.fileEncSha256,
          fileSha256: audio.fileSha256,
          fileLength: audio.fileLength
        },
        mimetype: audio.mimetype,
        ptt: audio.ptt || false
      };
      await sock.sendMessage(targetJid, content);
      return { ok: true };
    }

    if (msg.documentMessage) {
      const doc = msg.documentMessage;
      const content = {
        document: {
          url: doc.url,
          directPath: doc.directPath,
          mediaKey: doc.mediaKey,
          fileEncSha256: doc.fileEncSha256,
          fileSha256: doc.fileSha256,
          fileLength: doc.fileLength
        },
        mimetype: doc.mimetype,
        fileName: doc.fileName
      };
      if (doc.caption) content.caption = doc.caption;
      const mentions = doc.contextInfo?.mentionedJid || [];
      if (mentions.length > 0) content.mentions = mentions;
      await sock.sendMessage(targetJid, content);
      return { ok: true };
    }

    if (msg.stickerMessage) {
      const sticker = msg.stickerMessage;
      const content = {
        sticker: {
          url: sticker.url,
          directPath: sticker.directPath,
          mediaKey: sticker.mediaKey,
          fileEncSha256: sticker.fileEncSha256,
          fileSha256: sticker.fileSha256,
          fileLength: sticker.fileLength
        },
        mimetype: sticker.mimetype
      };
      await sock.sendMessage(targetJid, content);
      return { ok: true };
    }

    const msgType = Object.keys(msg)[0] || 'unknown';
    logger.warn({ targetJid, msgType }, 'reconstructAndSend: unsupported message type');
    return { ok: false, reason: 'unsupported' };
  } catch (err) {
    logger.warn({ err, targetJid }, 'reconstructAndSend: failed to send message');
    return { ok: false, reason: 'error' };
  }
}

async function fetchGroupJids(sock, chatId) {
  let groupJids = [];
  try {
    const groups = await sock.groupFetchAllParticipating();
    groupJids = Object.keys(groups || {});
  } catch (err) {
    logger.error({ err }, 'failed fetching groups for broadcast');
    try {
      await sock.sendMessage(chatId, { text: 'Failed to fetch group list.' });
    } catch (e) {
      logger.warn({ e }, 'failed sending group fetch error');
    }
    return null;
  }

  if (groupJids.length === 0) {
    try {
      await sock.sendMessage(chatId, { text: 'Bot is not in any groups.' });
    } catch (e) {
      logger.warn({ e }, 'failed sending no-groups message');
    }
    return null;
  }

  return groupJids;
}

async function handleBroadcastCommand({ chatId, senderId, text, quotedMessageId, contextMsgId, msg }) {
  const sock = getSock();
  if (!isOwnerJid(senderId)) {
    logger.info({ senderId, chatId }, '/broadcast rejected: not owner');
    try {
      await sock.sendMessage(chatId, { text: 'Only bot owners can use `/broadcast`.' });
    } catch (err) {
      logger.warn({ err }, 'failed sending broadcast rejection');
    }
    return;
  }

  const trimmedText = text && text.trim();
  const firstWord = trimmedText ? trimmedText.split(/\s+/)[0].toLowerCase() : '';
  const isDebug = firstWord === 'debug';
  const isTextBroadcast = trimmedText && !isDebug;

  if (isTextBroadcast) {
    // Text broadcast: /broadcast <text>
    const groupJids = await fetchGroupJids(sock, chatId);
    if (!groupJids) return;

    let sent = 0;
    let failed = 0;
    for (const groupJid of groupJids) {
      try {
        await sendRichMessage(sock, groupJid, { text: trimmedText, footer: 'Broadcast 📢', badge: false });
        sent += 1;
      } catch (err) {
        logger.warn({ err, groupJid }, 'broadcast send failed');
        failed += 1;
      }
    }

    try {
      const summary = `Broadcast complete: ${sent} group${sent !== 1 ? 's' : ''} sent${failed > 0 ? `, ${failed} failed` : ''}.`;
      await sock.sendMessage(chatId, { text: summary });
    } catch (err) {
      logger.warn({ err }, 'failed sending broadcast confirmation');
    }

    logger.info({ sent, failed, total: groupJids.length, chatId, senderId }, 'broadcast completed');
  } else if (isDebug && !quotedMessageId) {
    // Debug mode invoked without a quoted message
    try {
      await sock.sendMessage(chatId, { text: 'Reply to a message to use `/broadcast debug`.' });
    } catch (e) {
      logger.warn({ e }, 'failed sending debug no-reply error');
    }
  } else if (quotedMessageId) {
    // Reply broadcast: /broadcast (replying to a message), with optional 'debug' flag
    const cachedMsg = messageCache.get(quotedMessageId);
    if (!cachedMsg) {
      try {
        await sock.sendMessage(chatId, { text: 'Replied message not found in cache. Try replying to a more recent message.' });
      } catch (e) {
        logger.warn({ e }, 'failed sending cache-miss error');
      }
      return;
    }

    if (isDebug) {
      // Debug mode: send only to this chat
      const debugResult = await reconstructAndSend(sock, chatId, cachedMsg);
      try {
        if (debugResult.ok) {
          await sock.sendMessage(chatId, { text: 'Debug broadcast: message sent to this chat only.' });
        } else {
          await sock.sendMessage(chatId, { text: `Debug broadcast failed: ${debugResult.reason || 'unknown error'}.` });
        }
      } catch (err) {
        logger.warn({ err }, 'failed sending debug broadcast confirmation');
      }
      return;
    }

    // Full broadcast to all groups
    const groupJids = await fetchGroupJids(sock, chatId);
    if (!groupJids) return;

    let sent = 0;
    let failed = 0;
    let unsupported = 0;
    for (const groupJid of groupJids) {
      const result = await reconstructAndSend(sock, groupJid, cachedMsg);
      if (result.ok) {
        sent += 1;
      } else if (result.reason === 'unsupported') {
        unsupported += 1;
      } else {
        failed += 1;
      }
    }

    try {
      let summary = `Broadcast complete: ${sent} group${sent !== 1 ? 's' : ''} sent${failed > 0 ? `, ${failed} failed` : ''}`;
      if (unsupported > 0) {
        summary += ` (${unsupported} unsupported type${unsupported !== 1 ? 's' : ''})`;
      }
      summary += '.';
      await sock.sendMessage(chatId, { text: summary });
    } catch (err) {
      logger.warn({ err }, 'failed sending broadcast confirmation');
    }

    logger.info({ sent, failed, unsupported, total: groupJids.length, chatId, senderId }, 'broadcast completed');
  } else {
    try {
      await sock.sendMessage(chatId, { text: 'Usage: `/broadcast <text>`, or reply to a message with `/broadcast` (broadcasts to all groups) or `/broadcast debug` (sends only to this chat).' });
    } catch (e) {
      logger.warn({ e }, 'failed sending usage message');
    }
    return;
  }
}

export { handleBroadcastCommand };
