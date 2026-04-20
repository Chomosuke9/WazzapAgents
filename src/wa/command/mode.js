import { getSock } from '../connection.js';
import { getMode, setMode, getTriggers, VALID_MODES } from '../../db.js';

async function handleMode({ chatId, chatType, senderIsAdmin, senderIsOwner, senderId, args }) {
  const sock = getSock();

  if (chatType === 'private') {
    try {
      await sock.sendMessage(chatId, { text: '/mode can only be used in group chats.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (!senderIsOwner && !senderIsAdmin) {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins can change the mode.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (!args) {
    const current = getMode(chatId);
    const triggers = getTriggers(chatId);
    const triggersStr = triggers.size > 0 ? [...triggers].sort().join(', ') : 'none';
    try {
      await sock.sendMessage(chatId, {
        text: (
          `Current mode: *${current}*\n` +
          `Triggers (prefix/hybrid mode): ${triggersStr}\n\n` +
          '_auto_ = LLM1 decides when to respond\n' +
          '_prefix_ = only responds when tagged, replied, or name mentioned\n' +
          '_hybrid_ = checks prefix triggers first, falls back to auto (LLM1). If a prefix trigger arrives while LLM1 is running, LLM1 is cancelled and bot responds immediately'
        ),
      });
    } catch (err) { /* ignore */ }
    return;
  }

  const mode = args.trim().toLowerCase();
  if (!VALID_MODES.has(mode)) {
    try {
      await sock.sendMessage(chatId, { text: 'Invalid mode. Use: /mode auto, /mode prefix, or /mode hybrid' });
    } catch (err) { /* ignore */ }
    return;
  }

  setMode(chatId, mode);
  try {
    await sock.sendMessage(chatId, { text: `Mode updated: *${mode}*` });
  } catch (err) { /* ignore */ }
}

export { handleMode };
