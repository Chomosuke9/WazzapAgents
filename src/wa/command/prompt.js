import { getSock } from '../connection.js';
import wsClient from '../../wsClient.js';
import { getPrompt, setPrompt } from '../../db.js';

const PROMPT_MAX_CHARS = 4000;

async function handlePrompt({ chatId, chatType, senderIsAdmin, senderIsOwner, args }) {
  const sock = getSock();
  const isPrivate = chatType === 'private';

  if (isPrivate || senderIsOwner || senderIsAdmin) {
    // proceed
  } else {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins can use `/prompt`.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (!args) {
    const current = getPrompt(chatId);
    if (current) {
      try {
        await sock.sendMessage(chatId, { text: `Current prompt:\n${current}` });
      } catch (err) { /* ignore */ }
    } else {
      try {
        await sock.sendMessage(chatId, { text: 'No custom prompt set for this chat. Use `/prompt` <text> to set one.' });
      } catch (err) { /* ignore */ }
    }
    return;
  }

  if (args.trim().toLowerCase() === '-' || args.trim().toLowerCase() === 'clear' || args.trim().toLowerCase() === 'reset') {
    setPrompt(chatId, null);
    wsClient.sendReliable({ type: 'invalidate_chat_settings', chatId });
    try {
      await sock.sendMessage(chatId, { text: 'Custom prompt cleared. Bot will use the default.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (args.length > PROMPT_MAX_CHARS) {
    try {
      await sock.sendMessage(chatId, { text: `Prompt too long (${args.length} chars). Maximum is ${PROMPT_MAX_CHARS} characters.` });
    } catch (err) { /* ignore */ }
    return;
  }

  setPrompt(chatId, args);
  wsClient.sendReliable({ type: 'invalidate_chat_settings', chatId });
  const preview = args.length > 200 ? args.slice(0, 197) + '...' : args;
  try {
    await sock.sendMessage(chatId, { text: `Prompt updated:\n${preview}` });
  } catch (err) { /* ignore */ }
}

export { handlePrompt };
