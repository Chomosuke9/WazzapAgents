import logger from '../../logger.js';
import { isOwnerJid } from '../../participants.js';
import { messageCache } from '../../caches.js';
import { getSock } from '../connection.js';
import { sendRichMessage } from '../interactive/index.js';

async function handleBroadcastCommand({ chatId, senderId, text, quotedMessageId, contextMsgId, msg }) {
  const sock = getSock();
  if (!isOwnerJid(senderId)) {
    logger.info({ senderId, chatId }, '/broadcast rejected: not owner');
    try {
      await sock.sendMessage(chatId, { text: 'Only bot owners can use /broadcast.' });
    } catch (err) {
      logger.warn({ err }, 'failed sending broadcast rejection');
    }
    return;
  }

  // Collect all groups where bot is present
  let groupJids = [];
  try {
    const groups = await sock.groupFetchAllParticipating();
    groupJids = Object.keys(groups || {});
  } catch (err) {
    logger.error({ err }, 'failed fetching groups for broadcast');
    try {
      await sock.sendMessage(chatId, { text: 'Failed to fetch group list.' });
    } catch (e) { /* ignore */ }
    return;
  }

  if (groupJids.length === 0) {
    try {
      await sock.sendMessage(chatId, { text: 'Bot is not in any groups.' });
    } catch (e) { /* ignore */ }
    return;
  }

  let sent = 0;
  let failed = 0;

  if (text) {
    // Text broadcast: /broadcast <text>
    for (const groupJid of groupJids) {
      try {
        await sendRichMessage(sock, groupJid, { text, footer: 'Broadcast 📢', badge: false });
        sent += 1;
      } catch (err) {
        logger.warn({ err, groupJid }, 'broadcast send failed');
        failed += 1;
      }
    }
  } else if (quotedMessageId) {
    // Forward broadcast: /broadcast (replying to a message)
    const cachedMsg = messageCache.get(quotedMessageId);
    if (!cachedMsg) {
      try {
        await sock.sendMessage(chatId, { text: 'Replied message not found in cache. Try replying to a more recent message.' });
      } catch (e) { /* ignore */ }
      return;
    }

    for (const groupJid of groupJids) {
      try {
        // Use Baileys native forward
        await sock.sendMessage(groupJid, { 
          forward: cachedMsg,
          contextInfo: {
            isForwarded: true
          }
        });
        sent += 1;
      } catch (err) {
        logger.warn({ err, groupJid }, 'broadcast forward failed');
        failed += 1;
      }
    }
  } else {
    try {
      await sock.sendMessage(chatId, { text: 'Usage: /broadcast <text> or reply to a message with /broadcast.' });
    } catch (e) { /* ignore */ }
    return;
  }

  // Send confirmation
  try {
    const summary = `Broadcast complete: ${sent} group${sent !== 1 ? 's' : ''} sent${failed > 0 ? `, ${failed} failed` : ''}.`;
    await sock.sendMessage(chatId, { text: summary });
  } catch (err) {
    logger.warn({ err }, 'failed sending broadcast confirmation');
  }

  logger.info({ sent, failed, total: groupJids.length, chatId, senderId }, 'broadcast completed');
}

export { handleBroadcastCommand };
