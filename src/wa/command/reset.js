import logger from '../../logger.js';
import { getSock } from '../connection.js';
import wsClient from '../../wsClient.js';

async function handleReset({ chatId, chatType, senderIsAdmin, senderIsOwner, contextMsgId }) {
  const sock = getSock();
  const isPrivate = chatType === 'private';

  if (isPrivate || senderIsOwner || senderIsAdmin) {
    // proceed
  } else {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins can use /reset.' });
    } catch (err) { /* ignore */ }
    return;
  }

  wsClient.sendReliable({ type: 'clear_history', chatId });

  try {
    await sock.sendMessage(chatId, { text: 'Bot memory for this chat has been reset.' });
  } catch (err) { /* ignore */ }

  logger.info({ chatId }, 'Memory cleared via /reset');
}

export { handleReset };
