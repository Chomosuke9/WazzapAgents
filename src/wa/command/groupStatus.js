import path from "path";
import fs from "fs-extra";
import {
  generateWAMessageContent,
  generateWAMessageFromContent,
} from "baileys";
import logger from "../../logger.js";
import { getSock } from "../connection.js";
import { extractContextInfo } from "../../messageParser.js";
import { findRawMediaContent } from "./groupStatusHelpers.js";
import { downloadMediaToFile, mapMediaKind } from "../../mediaHandler.js";
import config from "../../config.js";
import { withTimeout } from "../utils.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Strip device ID from a JID to get the canonical form.
 * e.g. "62812xxx:5@s.whatsapp.net" -> "62812xxx@s.whatsapp.net"
 */
function getCleanJid(jid) {
  return jid.split(":")[0].split("/")[0] + "@s.whatsapp.net";
}

/**
 * Build the contextInfo required for group status messages.
 * - isGroupStatus: true  — marks this as a group status
 * - statusAttributions  — carries the type + authorJid so WhatsApp shows attribution
 */
function createGroupStatusContextInfo(authorJid) {
  return {
    isGroupStatus: true,
    statusAttributions: [
      {
        type: 5, // StatusAttributionType.GROUP_STATUS
        groupStatus: { authorJid: getCleanJid(authorJid) },
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Media download helper
// ---------------------------------------------------------------------------

async function downloadMediaContent(content, contentType, messageId) {
  const mediaKind = mapMediaKind(contentType);
  if (!mediaKind || !["image", "video"].includes(mediaKind)) return null;

  try {
    const extMap = { image: "jpg", video: "mp4" };
    const ext = extMap[mediaKind] || "bin";
    const filename = `${messageId}_groupStatus.${ext}`;
    const filepath = path.join(config.mediaDir, filename);
    await downloadMediaToFile(content, mediaKind, filepath, withTimeout);
    return { filepath, mediaKind };
  } catch (err) {
    logger.warn(
      { err, messageId, contentType },
      "failed to download media for group-status",
    );
    return null;
  }
}

// ---------------------------------------------------------------------------
// Send helper — unified group status via groupStatusMessageV2
// ---------------------------------------------------------------------------

/**
 * Send a group status message (text, image, or video) to a group.
 *
 * Uses groupStatusMessageV2 (FutureProofMessage wrapper, proto field tag 826)
 * which is the newer version replacing groupStatusMessage (tag 770).
 *
 * Flow:
 *   1. generateWAMessageContent — uploads media (if any) and builds the proto
 *   2. Inject contextInfo (isGroupStatus + statusAttributions) for attribution
 *   3. Add messageContextInfo with device metadata
 *   4. Wrap inside groupStatusMessageV2 FutureProofMessage
 *   5. generateWAMessageFromContent + relayMessage (same pattern as sendInteractive.js)
 *
 * @param {object} sock - Baileys socket instance
 * @param {string} jid - Target group JID (e.g., '120363xxx@g.us')
 * @param {object} content - { text } | { image: { url }, caption? } | { video: { url }, caption? }
 * @param {string} authorJid - Bot's JID for status attribution
 * @returns {Promise<object>} Generated message object
 */
async function sendGroupStatus(sock, jid, content, authorJid) {
  // Step 1 — generate message content (handles text, image, video uniformly)
  const waMsgContent = await generateWAMessageContent(content, {
    upload: sock.waUploadToServer,
  });

  // generateWAMessageContent may wrap the inner Message in a .message field
  const innerMsg = waMsgContent.message || waMsgContent;

  // Step 2 — inject contextInfo for attribution into applicable message types
  const contextInfo = createGroupStatusContextInfo(authorJid);
  for (const key of Object.keys(innerMsg)) {
    if (
      key === "imageMessage" ||
      key === "videoMessage" ||
      key === "extendedTextMessage"
    ) {
      innerMsg[key].contextInfo = contextInfo;
    }
  }

  // Step 3 — add messageContextInfo with device metadata
  innerMsg.messageContextInfo = {
    deviceListMetadata: {},
    deviceListMetadataVersion: 2,
  };

  // Step 4 — wrap in groupStatusMessageV2 (FutureProofMessage)
  const wrapper = { groupStatusMessageV2: { message: innerMsg } };

  // Step 5 — generate full message and relay (same pattern as sendInteractive.js)
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

async function handleGroupStatus({
  chatId,
  chatType,
  senderIsAdmin,
  senderIsOwner,
  senderId,
  args,
  msg,
  fromMe,
}) {
  const sock = getSock();

  // Only works in groups
  if (chatType !== "group") {
    try {
      await sock.sendMessage(chatId, {
        text: "This command can only be used in a group.",
      });
    } catch (err) {
      /* ignore */
    }
    return;
  }

  // Permission: admin, owner, or the bot itself (fromMe)
  if (!senderIsAdmin && !senderIsOwner && !fromMe) {
    logger.info({ chatId }, "/group-status rejected: not admin, owner, or bot");
    try {
      await sock.sendMessage(chatId, {
        text: "Only admin/owner can send group status.",
      });
    } catch (err) {
      /* ignore */
    }
    return;
  }

  const caption = (args || "").trim();
  const authorJid = sock.user?.id || senderId;
  let mediaResult = null;

  // Mode 1: Media attached directly to this message
  // Use raw key inspection (NOT normalizeMessageContent) because
  // normalizeMessageContent transforms imageMessage-with-caption into
  // extendedTextMessage, destroying the downloadable media object.
  const rawMedia = findRawMediaContent(msg.message);
  if (rawMedia) {
    mediaResult = await downloadMediaContent(
      rawMedia.content,
      rawMedia.contentType,
      msg.key.id,
    );
  }

  // Mode 2: Reply to a media message
  // Use extractContextInfo (not innerMessage?.extendedTextMessage?.contextInfo)
  // so contextInfo is found regardless of the outer message type.
  // Use findRawMediaContent on the quoted message (NOT unwrapMessage) for the
  // same reason as Mode 1 — unwrapMessage calls normalizeMessageContent which
  // transforms imageMessage-with-caption into extendedTextMessage, silently
  // dropping the downloadable media object.
  if (!mediaResult) {
    const ctx = extractContextInfo(msg.message);
    if (ctx?.quotedMessage) {
      const quotedMedia = findRawMediaContent(ctx.quotedMessage);
      if (quotedMedia) {
        mediaResult = await downloadMediaContent(
          quotedMedia.content,
          quotedMedia.contentType,
          ctx.stanzaId,
        );
      }
    }
  }

  try {
    if (mediaResult) {
      const content = {
        [mediaResult.mediaKind]: { url: mediaResult.filepath },
        caption: caption || "",
      };
      await sendGroupStatus(sock, chatId, content, authorJid);
      logger.info(
        { chatId, mediaKind: mediaResult.mediaKind, hasCaption: !!caption },
        "group-status sent with media",
      );
    } else if (caption) {
      await sendGroupStatus(sock, chatId, { text: caption }, authorJid);
      logger.info({ chatId }, "group-status sent as text");
    } else {
      try {
        await sock.sendMessage(chatId, {
          text: "Reply to an image/video or provide text.",
        });
      } catch (err) {
        /* ignore */
      }
      return;
    }
  } catch (err) {
    logger.error({ err, chatId }, "failed to send group-status");
    try {
      await sock.sendMessage(chatId, {
        text: `Failed to send group status: ${err.message}`,
      });
    } catch (e) {
      /* ignore */
    }
  } finally {
    if (mediaResult?.filepath) {
      try {
        await fs.remove(mediaResult.filepath);
      } catch {
        /* ignore */
      }
    }
  }
}

export { handleGroupStatus, sendGroupStatus };
