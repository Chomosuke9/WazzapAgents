import logger from '../../logger.js';
import { isOwnerJid } from '../../participants.js';
import { messageCache } from '../../caches.js';
import { getSock } from '../connection.js';
import { sendRichMessage } from '../interactive/index.js';

async function reconstructAndSend(sock, targetJid, cachedMsg) {
  const msg = cachedMsg.message;
  if (!msg) {
    logger.warn({ targetJid }, 'reconstructAndSend: cachedMsg.message is empty');
    return false;
  }

  try {
    if (msg.conversation || msg.extendedTextMessage) {
      const text = msg.conversation || msg.extendedTextMessage?.text;
      const mentions = msg.extendedTextMessage?.contextInfo?.mentionedJid || [];
      const content = { text };
      if (mentions.length > 0) content.mentions = mentions;
      await sock.sendMessage(targetJid, content);
      return true;
    }

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
      return true;
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
      return true;
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
      return true;
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
      return true;
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
      return true;
    }

    const msgType = Object.keys(msg)[0] || 'unknown';
    logger.warn({ targetJid, msgType }, 'reconstructAndSend: unsupported message type');
    return false;
  } catch (err) {
    logger.warn({ err, targetJid }, 'reconstructAndSend: failed to send message');
    return false;
  }
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
  const isTextBroadcast = trimmedText && trimmedText.toLowerCase() !== 'debug';
  const isDebug = trimmedText && trimmedText.toLowerCase() === 'debug';

  if (isTextBroadcast) {
    // Text broadcast: /broadcast <text>
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
      return;
    }

    if (groupJids.length === 0) {
      try {
        await sock.sendMessage(chatId, { text: 'Bot is not in any groups.' });
      } catch (e) {
        logger.warn({ e }, 'failed sending no-groups message');
      }
      return;
    }

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
      await reconstructAndSend(sock, chatId, cachedMsg);
      try {
        await sock.sendMessage(chatId, { text: 'Debug broadcast: message sent to this chat only.' });
      } catch (err) {
        logger.warn({ err }, 'failed sending debug broadcast confirmation');
      }
      return;
    }

    // Full broadcast to all groups
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
      return;
    }

    if (groupJids.length === 0) {
      try {
        await sock.sendMessage(chatId, { text: 'Bot is not in any groups.' });
      } catch (e) {
        logger.warn({ e }, 'failed sending no-groups message');
      }
      return;
    }

    let sent = 0;
    let failed = 0;
    for (const groupJid of groupJids) {
      const ok = await reconstructAndSend(sock, groupJid, cachedMsg);
      if (ok) {
        sent += 1;
      } else {
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
