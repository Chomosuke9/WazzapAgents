import logger from '../../logger.js';
import { getSock } from '../connection.js';
import { sendOutgoing } from '../outbound.js';
import { createStickerFile } from '../stickerTool.js';
import { unwrapMessage } from '../../messageParser.js';
import { downloadMediaToFile, mapMediaKind } from '../../mediaHandler.js';
import config from '../../config.js';
import { withTimeout } from '../utils.js';

function parseStickerArgs(args) {
  if (!args || !args.trim()) return [null, null];
  if (args.includes('#')) {
    const [upper, lower] = args.split('#');
    return [upper.trim() || null, lower.trim() || null];
  }
  return [args.trim() || null, null];
}

async function downloadMediaContent(content, contentType, messageId) {
  const mediaKind = mapMediaKind(contentType);
  if (!mediaKind || mediaKind !== 'image') return null;

  try {
    const ext = mediaKind === 'video' ? 'mp4' : 'jpg';
    const filename = `${messageId}_${mediaKind}.${ext}`;
    const filepath = `${config.mediaDir}/${filename}`;
    await downloadMediaToFile(content, mediaKind, filepath, withTimeout);
    return filepath;
  } catch (err) {
    logger.warn({ err, messageId, contentType }, 'failed to download media for sticker');
    return null;
  }
}

async function handleSticker({ chatId, chatType, senderIsAdmin, senderIsOwner, args, msg }) {
  const [upperText, lowerText] = parseStickerArgs(args);

  const { contentType, message: innerMessage } = unwrapMessage(msg.message) || {};
  let mediaPath = null;

  if (contentType === 'imageMessage') {
    mediaPath = await downloadMediaContent(innerMessage[contentType], contentType, msg.key.id);
  } else if (contentType === 'videoMessage') {
    try {
      await getSock().sendMessage(chatId, { text: '/sticker saat ini hanya mendukung gambar (video belum didukung).' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (!mediaPath && innerMessage?.extendedTextMessage?.contextInfo) {
    const ctx = innerMessage.extendedTextMessage.contextInfo;
    if (ctx.quotedMessage) {
      const { contentType: qType, message: qMsg } = unwrapMessage(ctx.quotedMessage) || {};
      const qContent = qType ? qMsg?.[qType] : null;
      if (qType === 'imageMessage') {
        mediaPath = await downloadMediaContent(qContent, qType, ctx.stanzaId);
      } else if (qType === 'videoMessage') {
        try {
          await getSock().sendMessage(chatId, { text: '/sticker saat ini hanya mendukung gambar (video belum didukung).' });
        } catch (err) { /* ignore */ }
        return;
      }
    }
  }

  if (!mediaPath) {
    try {
      await getSock().sendMessage(chatId, { text: 'Send an image with /sticker caption, or reply to an image.' });
    } catch (err) { /* ignore */ }
    return;
  }

  try {
    const stickerPath = await createStickerFile(mediaPath, upperText, lowerText);
    await sendOutgoing({
      chatId,
      attachments: [{ kind: 'sticker', path: stickerPath }],
      replyTo: msg.key.id,
    });
    logger.info({ chatId }, 'Sticker created and sent');
  } catch (err) {
    logger.error({ err, chatId }, 'failed to create sticker');
    try {
      await getSock().sendMessage(chatId, { text: `Failed to create sticker: ${err.message}` });
    } catch (e) { /* ignore */ }
  }
}

export { handleSticker };
