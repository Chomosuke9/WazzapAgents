import { generateWAMessageFromContent } from "baileys";
import logger from "../../logger.js";

/**
 * Internal helper that wraps an interactiveMessage payload with the required
 * Protobuf structure (viewOnceMessage + deviceListMetadata) and relays it.
 *
 * @param {object} sock - Baileys socket instance
 * @param {string} jid - Recipient JID
 * @param {object} interactiveContent - Raw interactiveMessage content
 * @returns {Promise<object>} Generated Baileys message object (includes key.id)
 */
async function sendInteractive(sock, jid, interactiveContent) {
  const content = {
    messageContextInfo: {
      deviceListMetadata: {},
      deviceListMetadataVersion: 2,
      forwardedNewsletterMessageInfo: {
        newsletterJid: "0@newsletter",
        serverMessageId: -1,
        newsletterName: "WazzapAgents",
      },
    },
    interactiveMessage: interactiveContent,
  };

  const msg = generateWAMessageFromContent(jid, content, {
    userJid: sock.user.id,
  });

  logger.debug({ jid, messageId: msg.key.id }, "relaying interactive message");
  await sock.relayMessage(jid, msg.message, { messageId: msg.key.id });

  return msg;
}

export { sendInteractive };
