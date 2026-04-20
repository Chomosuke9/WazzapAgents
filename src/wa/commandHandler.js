import { unwrapMessage } from '../messageParser.js';
import {
  handleBroadcastCommand,
  handleInfoCommand,
  handleDebugCommand,
  handleJoinCommand,
  handleHelp,
  handlePrompt,
  handleReset,
  handleSticker,
  handlePermission,
  handleMode,
  handleTrigger,
  handleDashboard,
  handleModel,
  handleModelcfg,
  handleSettings,
  handleGroupStatus,
} from './command/index.js';

async function handleCommandListener(msg, context) {
  const { slashCommand, chatId, chatType, senderIsAdmin, senderId, botIsAdmin, senderIsOwner, contextMsgId } = context;

  if (!slashCommand) return false;

  const { command, args } = slashCommand;

  // Extract quoted message ID if any
  const { message: innerMessage } = unwrapMessage(msg.message);
  const quotedMessageId = innerMessage?.extendedTextMessage?.contextInfo?.stanzaId || null;

  switch (command) {
    case 'help':
      await handleHelp({ chatId });
      return true;

    case 'prompt':
      await handlePrompt({ chatId, chatType, senderIsAdmin, senderIsOwner, args });
      return true;

    case 'reset':
      await handleReset({ chatId, chatType, senderIsAdmin, senderIsOwner, contextMsgId });
      return true;

    case 'permission':
      await handlePermission({ chatId, chatType, senderIsAdmin, senderIsOwner, botIsAdmin, args });
      return true;

    case 'mode':
      await handleMode({ chatId, chatType, senderIsAdmin, senderIsOwner, senderId, args });
      return true;

    case 'trigger':
      await handleTrigger({ chatId, chatType, senderIsAdmin, senderIsOwner, senderId, args });
      return true;

    case 'dashboard':
      await handleDashboard({ chatId });
      return true;

    case 'broadcast':
      await handleBroadcastCommand({
        chatId,
        senderId,
        text: args,
        quotedMessageId,
        contextMsgId,
        msg,
      });
      return true;

    case 'info':
      await handleInfoCommand({
        chatId,
        senderId,
        senderDisplay: context.senderDisplay,
        senderRole: context.senderRole,
        isGroup: chatType === 'group',
        group: context.group,
      });
      return true;

    case 'debug':
      await handleDebugCommand({ chatId, senderId, args });
      return true;

    case 'join':
      await handleJoinCommand({ chatId, senderId, args });
      return true;

    case 'sticker':
      await handleSticker({ chatId, chatType, senderIsAdmin, senderIsOwner, args, msg });
      return true;

    case 'model':
      await handleModel({ chatId, chatType, senderIsAdmin, senderIsOwner, args });
      return true;

    case 'modelcfg':
      await handleModelcfg({ chatId, senderId, senderIsOwner, args });
      return true;

    case 'setting':
      await handleSettings({ chatId, chatType, senderId, senderIsAdmin, senderIsOwner, args });
      return true;

    case 'group-status':
      await handleGroupStatus({ chatId, chatType, senderIsAdmin, senderIsOwner, args, msg });
      return true;

    default:
      return false;
  }
}

export { handleCommandListener };
