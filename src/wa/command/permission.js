import { getSock } from '../connection.js';
import { getPermission, setPermission } from '../../db.js';

const PERMISSION_LABELS = {
  0: '0 (all moderation forbidden)',
  1: '1 (delete allowed)',
  2: '2 (delete & mute allowed)',
  3: '3 (delete, mute & kick allowed)',
};

async function handlePermission({ chatId, chatType, senderIsAdmin, senderIsOwner, botIsAdmin, args }) {
  const sock = getSock();

  if (chatType === 'private') {
    try {
      await sock.sendMessage(chatId, { text: '/permission can only be used in group chats.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (!senderIsOwner && !senderIsAdmin) {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins or bot owner can use /permission.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (!args) {
    const current = getPermission(chatId);
    const label = PERMISSION_LABELS[current] || String(current);
    try {
      await sock.sendMessage(chatId, { text: `Current permission level: ${label}` });
    } catch (err) { /* ignore */ }
    return;
  }

  const level = parseInt(args.trim(), 10);
  if (isNaN(level)) {
    try {
      await sock.sendMessage(chatId, { text: 'Usage: /permission 0, 1, 2, or 3.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (level < 0 || level > 3) {
    try {
      await sock.sendMessage(chatId, { text: 'Level must be 0-3.\n0: all forbidden\n1: delete\n2: delete & mute\n3: delete, mute & kick' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (level > 0 && !botIsAdmin) {
    try {
      await sock.sendMessage(chatId, { text: 'Bot must be an admin to enable moderation (permission 1-3). Promote the bot first, then try again.' });
    } catch (err) { /* ignore */ }
    return;
  }

  setPermission(chatId, level);
  const label = PERMISSION_LABELS[level] || String(level);
  try {
    await sock.sendMessage(chatId, { text: `Permission updated: ${label}` });
  } catch (err) { /* ignore */ }
}

export { handlePermission };
