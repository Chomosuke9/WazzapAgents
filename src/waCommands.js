import logger from './logger.js';
import { isOwnerJid } from './participants.js';
import { messageCache } from './caches.js';
import { getSock } from './waConnection.js';

// ---------------------------------------------------------------------------
// Slash command parsing
// ---------------------------------------------------------------------------

const SLASH_CMD_RE = /^\/(broadcast|prompt|reset|permission|info|mode|trigger|dashboard|help)\b\s*([\s\S]*)/i;

function parseSlashCommand(text) {
  if (!text || typeof text !== 'string') return null;
  const m = text.trim().match(SLASH_CMD_RE);
  if (!m) return null;
  return {
    command: m[1].toLowerCase(),
    args: (m[2] || '').trim(),
  };
}

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
        await sock.sendMessage(groupJid, { text });
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
        await sock.sendMessage(groupJid, { forward: cachedMsg });
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

function truncateText(value, maxChars = 300) {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (trimmed.length <= maxChars) return trimmed;
  return `${trimmed.slice(0, Math.max(0, maxChars - 3))}...`;
}

async function handleInfoCommand({ chatId, senderId, senderDisplay, senderRole, isGroup, group }) {
  const sock = getSock();
  const isOwner = isOwnerJid(senderId);
  const roleLabel = isOwner
    ? 'owner'
    : (senderRole?.isSuperAdmin ? 'superadmin' : (senderRole?.isAdmin ? 'admin' : 'member'));
  const lines = [
    'Info pengguna:',
    `Nama: ${senderDisplay || 'unknown'}`,
    `JID: ${senderId || 'unknown'}`,
    `Peran: ${roleLabel}`,
    `Owner bot: ${isOwner ? 'ya' : 'tidak'}`,
  ];

  if (isGroup) {
    const groupName = group?.name || chatId;
    const memberCount = Array.isArray(group?.participants) ? group.participants.length : null;
    const description = truncateText(group?.description, 300);
    lines.push('');
    lines.push('Info grup:');
    lines.push(`Nama grup: ${groupName || 'unknown'}`);
    lines.push(`ID grup: ${chatId || 'unknown'}`);
    lines.push(`Jumlah anggota: ${typeof memberCount === 'number' ? memberCount : 'unknown'}`);
    lines.push(`Bot admin: ${group?.botIsAdmin ? 'ya' : 'tidak'}`);
    lines.push(`Bot superadmin: ${group?.botIsSuperAdmin ? 'ya' : 'tidak'}`);
    if (description) lines.push(`Deskripsi: ${description}`);
  } else {
    lines.push('');
    lines.push('Info chat:');
    lines.push('Tipe: private');
    lines.push(`ID chat: ${chatId || 'unknown'}`);
  }

  try {
    await sock.sendMessage(chatId, { text: lines.join('\n') });
  } catch (err) {
    logger.warn({ err, chatId }, 'failed sending /info response');
  }
}

export {
  parseSlashCommand,
  handleBroadcastCommand,
  truncateText,
  handleInfoCommand,
};
