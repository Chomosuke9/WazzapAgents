import path from 'path';
import fs from 'fs-extra';
import { proto, generateWAMessageContent, generateWAMessageFromContent } from 'baileys';
import logger from '../../logger.js';
import { getSock } from '../connection.js';
import { unwrapMessage } from '../../messageParser.js';
import { downloadMediaToFile, mapMediaKind } from '../../mediaHandler.js';
import config from '../../config.js';
import { withTimeout } from '../utils.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Strip device ID from a JID to get the canonical form.
 * e.g. "62812xxx:5@s.whatsapp.net" -> "62812xxx@s.whatsapp.net"
 */
function getCleanJid(jid) {
  return jid.split(':')[0].split('/')[0] + '@s.whatsapp.net';
}

/**
 * Wrap an inner Message inside a groupStatusMessage (FutureProofMessage)
 * so it renders as a Group Status inside the group chat.
 */
function createGroupStatusMessage(innerMessage) {
  return { groupStatusMessage: { message: innerMessage } };
}

/**
 * Build the contextInfo required for group status messages.
 * - isGroupStatus: true  — marks this as a group status
 * - statusAttributions  — carries the authorJid so WhatsApp shows attribution
 */
function createGroupStatusContextInfo(authorJid) {
  return {
    isGroupStatus: true,
    statusAttributions: [{
      groupStatus: { authorJid: getCleanJid(authorJid) },
    }],
  };
}

// ---------------------------------------------------------------------------
// Media download helper
// ---------------------------------------------------------------------------

async function downloadMediaContent(content, contentType, messageId) {
  const mediaKind = mapMediaKind(contentType);
  if (!mediaKind || !['image', 'video'].includes(mediaKind)) return null;

  try {
    const extMap = { image: 'jpg', video: 'mp4' };
    const ext = extMap[mediaKind] || 'bin';
    const filename = `${messageId}_groupStatus.${ext}`;
    const filepath = path.join(config.mediaDir, filename);
    await downloadMediaToFile(content, mediaKind, filepath, withTimeout);
    return { filepath, mediaKind };
  } catch (err) {
    logger.warn({ err, messageId, contentType }, 'failed to download media for group-status');
    return null;
  }
}

// ---------------------------------------------------------------------------
// Send helpers — relayMessage + groupStatusMessage proto wrapper
// ---------------------------------------------------------------------------

/**
 * Send a text-only group status message.
 *
 * Structure:
 *   groupStatusMessage.message.extendedTextMessage
 *     .text = caption
 *     .contextInfo = { isGroupStatus: true, statusAttributions: [...] }
 */
async function sendTextGroupStatus(sock, jid, text, authorJid) {
  const contextInfo = createGroupStatusContextInfo(authorJid);
  const innerMessage = {
    extendedTextMessage: {
      text,
      contextInfo,
    },
  };

  const wrapper = createGroupStatusMessage(innerMessage);

  const msg = generateWAMessageFromContent(jid, wrapper, {
    userJid: sock.user.id,
  });

  await sock.relayMessage(jid, msg.message, {
    messageId: msg.key.id,
  });

  return msg;
}

/**
 * Send an image or video group status message.
 *
 * 1. Upload media via generateWAMessageContent (uses sock.waUploadToServer)
 * 2. Inject contextInfo (isGroupStatus + statusAttributions) into the
 *    resulting imageMessage/videoMessage proto
 * 3. Wrap inside groupStatusMessage FutureProofMessage
 * 4. Relay via relayMessage
 */
async function sendMediaGroupStatus(sock, jid, mediaPath, mediaKind, caption, authorJid) {
  const contextInfo = createGroupStatusContextInfo(authorJid);

  // Step 1 — upload media and get the proto object
  const contentKey = mediaKind; // 'image' or 'video'
  const uploaded = await generateWAMessageContent({
    [contentKey]: { url: mediaPath },
    caption: caption || '',
  }, {
    upload: sock.waUploadToServer,
  });

  // Step 2 — inject contextInfo into the media message
  const msgKey = mediaKind === 'image' ? 'imageMessage' : 'videoMessage';
  if (uploaded[msgKey]) {
    uploaded[msgKey].contextInfo = contextInfo;
    if (caption) {
      uploaded[msgKey].caption = caption;
    }
  }

  // Step 3 — wrap in groupStatusMessage
  const wrapper = createGroupStatusMessage(uploaded);

  // Step 4 — generate and relay
  const msg = generateWAMessageFromContent(jid, wrapper, {
    userJid: sock.user.id,
  });

  await sock.relayMessage(jid, msg.message, {
    messageId: msg.key.id,
  });

  return msg;
}

// ---------------------------------------------------------------------------
// Command handler
// ---------------------------------------------------------------------------

async function handleGroupStatus({ chatId, chatType, senderIsAdmin, senderIsOwner, senderId, args, msg }) {
  const sock = getSock();

  // Only works in groups
  if (chatType !== 'group') {
    try {
      await sock.sendMessage(chatId, { text: 'This command can only be used in a group.' });
    } catch (err) { /* ignore */ }
    return;
  }

  // Permission: admin or owner only
  if (!senderIsAdmin && !senderIsOwner) {
    logger.info({ chatId }, '/group-status rejected: not admin or owner');
    try {
      await sock.sendMessage(chatId, { text: 'Only admin/owner can send group status.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const caption = (args || '').trim();
  const authorJid = sock.user?.id || senderId;
  const { contentType, message: innerMessage } = unwrapMessage(msg.message) || {};
  let mediaResult = null;

  // Mode 1: Media attached directly to this message
  if (contentType === 'imageMessage' || contentType === 'videoMessage') {
    mediaResult = await downloadMediaContent(innerMessage[contentType], contentType, msg.key.id);
  }

  // Mode 2: Reply to a media message
  if (!mediaResult && innerMessage?.extendedTextMessage?.contextInfo) {
    const ctx = innerMessage.extendedTextMessage.contextInfo;
    if (ctx.quotedMessage) {
      const { contentType: qType, message: qMsg } = unwrapMessage(ctx.quotedMessage) || {};
      if ((qType === 'imageMessage' || qType === 'videoMessage') && qMsg?.[qType]) {
        mediaResult = await downloadMediaContent(qMsg[qType], qType, ctx.stanzaId);
      }
    }
  }

  try {
    if (mediaResult) {
      await sendMediaGroupStatus(sock, chatId, mediaResult.filepath, mediaResult.mediaKind, caption, authorJid);
      logger.info({ chatId, mediaKind: mediaResult.mediaKind, hasCaption: !!caption }, 'group-status sent with media');
    } else if (caption) {
      await sendTextGroupStatus(sock, chatId, caption, authorJid);
      logger.info({ chatId }, 'group-status sent as text');
    } else {
      try {
        await sock.sendMessage(chatId, {
          text: 'Reply to an image/video or provide text.',
        });
      } catch (err) { /* ignore */ }
      return;
    }
  } catch (err) {
    logger.error({ err, chatId }, 'failed to send group-status');
    try {
      await sock.sendMessage(chatId, { text: `Failed to send group status: ${err.message}` });
    } catch (e) { /* ignore */ }
  } finally {
    if (mediaResult?.filepath) {
      try { await fs.remove(mediaResult.filepath); } catch { /* ignore */ }
    }
  }
}

export { handleGroupStatus };
