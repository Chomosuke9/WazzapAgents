import { getSock } from '../connection.js';
import { getSubagentEnabled, setSubagentEnabled } from '../../db.js';

async function handleSubagent({ chatId, senderIsOwner, args }) {
  const sock = getSock();

  if (!senderIsOwner) {
    try {
      await sock.sendMessage(chatId, { text: 'Only bot owner can use `/subagent`.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (!args) {
    const current = getSubagentEnabled(chatId);
    try {
      await sock.sendMessage(chatId, {
        text: `Subagent: *${current ? 'ON' : 'OFF'}*\n\n` +
          'Enable subagent for this chat to allow LLM2 to call sub-agents for complex tasks.\n\n' +
          '_/subagent on_ - enable subagent\n' +
          '_/subagent off_ - disable subagent',
      });
    } catch (err) { /* ignore */ }
    return;
  }

  const value = args.trim().toLowerCase();
  if (value === 'on') {
    setSubagentEnabled(chatId, true);
    try {
      await sock.sendMessage(chatId, { text: 'Subagent enabled.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (value === 'off') {
    setSubagentEnabled(chatId, false);
    try {
      await sock.sendMessage(chatId, { text: 'Subagent disabled.' });
    } catch (err) { /* ignore */ }
    return;
  }

  try {
    await sock.sendMessage(chatId, { text: 'Invalid. Use `/subagent on` or `/subagent off`' });
  } catch (err) { /* ignore */ }
}

export { handleSubagent };