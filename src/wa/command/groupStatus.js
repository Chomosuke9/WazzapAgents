import path from 'path';
import fs from 'fs-extra';
import logger from '../../logger.js';
import { getSock } from '../connection.js';
import { unwrapMessage } from '../../messageParser.js';
import { downloadMediaToFile, mapMediaKind } from '../../mediaHandler.js';
import config from '../../config.js';
import { withTimeout } from '../utils.js';

// ---------------------------------------------------------------------------
// Media download helper
// ---------------------------------------------------------------------------

async function downloadImageContent(content, contentType, messageId) {
  const mediaKind = mapMediaKind(contentType);
  if (!mediaKind || mediaKind !== 'image') return null;

  try {
    const filename = `${messageId}_groupStatus.jpg`;
    const filepath = path.join(config.mediaDir, filename);
    await downloadMediaToFile(content, mediaKind, filepath, withTimeout);
    return filepath;
  } catch (err) {
    logger.warn({ err, messageId, contentType }, 'failed to download media for group-status');
    return null;
  }
}

// ---------------------------------------------------------------------------
// Command handler
// ---------------------------------------------------------------------------

async function handleGroupStatus({ chatId, chatType, senderIsAdmin, senderIsOwner, args, msg }) {
  const sock = getSock();

  // Only works in groups
  if (chatType !== 'group') {
    try {
      await sock.sendMessage(chatId, { text: 'Perintah `/group-status` hanya bisa digunakan di grup.' });
    } catch (err) { /* ignore */ }
    return;
  }

  // Permission: admin or owner only
  if (!senderIsAdmin && !senderIsOwner) {
    logger.info({ chatId }, '/group-status rejected: not admin or owner');
    try {
      await sock.sendMessage(chatId, { text: 'Hanya admin grup atau owner bot yang bisa menggunakan `/group-status`.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const caption = (args || '').trim();
  const { contentType, message: innerMessage } = unwrapMessage(msg.message) || {};
  let mediaPath = null;

  // Mode 1: Image attached directly to this message (e.g. send image with caption /group-status <text>)
  if (contentType === 'imageMessage') {
    mediaPath = await downloadImageContent(innerMessage[contentType], contentType, msg.key.id);
  }

  // Mode 2: Reply to an image message
  if (!mediaPath && innerMessage?.extendedTextMessage?.contextInfo) {
    const ctx = innerMessage.extendedTextMessage.contextInfo;
    if (ctx.quotedMessage) {
      const { contentType: qType, message: qMsg } = unwrapMessage(ctx.quotedMessage) || {};
      if (qType === 'imageMessage' && qMsg?.[qType]) {
        mediaPath = await downloadImageContent(qMsg[qType], qType, ctx.stanzaId);
      }
    }
  }

  try {
    if (mediaPath) {
      // Send image as group status
      await sock.sendMessage(chatId, {
        image: { url: mediaPath },
        caption: caption || '',
        groupStatus: true,
      });
      logger.info({ chatId, hasCaption: !!caption }, 'group-status sent with image');
    } else if (caption) {
      // Text-only group status
      await sock.sendMessage(chatId, {
        text: caption,
        groupStatus: true,
      });
      logger.info({ chatId }, 'group-status sent as text');
    } else {
      // No image and no caption
      try {
        await sock.sendMessage(chatId, {
          text: 'Penggunaan: kirim gambar dengan caption `/group-status <teks>`, reply gambar dengan `/group-status <teks>`, atau `/group-status <teks>` untuk status teks saja.',
        });
      } catch (err) { /* ignore */ }
      return;
    }
  } catch (err) {
    logger.error({ err, chatId }, 'failed to send group-status');
    try {
      await sock.sendMessage(chatId, { text: `Gagal mengirim group status: ${err.message}` });
    } catch (e) { /* ignore */ }
  } finally {
    // Cleanup downloaded media file
    if (mediaPath) {
      try { await fs.remove(mediaPath); } catch { /* ignore */ }
    }
  }
}

export { handleGroupStatus };
