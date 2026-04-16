import logger from '../logger.js';
import { isOwnerJid, roleFlagsForJid } from '../participants.js';
import { getSock } from './connection.js';
import { sendRichMessage } from './interactive/index.js';
import {
  getPrompt,
  setPrompt,
  getPermission,
  setPermission,
  getMode,
  setMode,
  getTriggers,
  setTriggers,
  clearSettings,
  getStats,
  getTopUsers,
  VALID_MODES,
  VALID_TRIGGERS,
} from '../db.js';
import wsClient from '../wsClient.js';
import {
  handleBroadcastCommand,
  handleInfoCommand,
  handleDebugCommand,
  handleJoinCommand,
} from './commands.js';

const PROMPT_MAX_CHARS = 4000;

const PERMISSION_LABELS = {
  0: '0 (all moderation forbidden)',
  1: '1 (delete allowed)',
  2: '2 (delete & mute allowed)',
  3: '3 (delete, mute & kick allowed)',
};

const TRIGGER_DESCRIPTIONS = {
  tag: 'bot @mentioned',
  reply: 'replied to bot message',
  join: 'new member joins group',
  name: 'bot name mentioned in text',
};

const HELP_TEXT = `*Daftar Perintah Bot*

*/prompt* [teks] — atur kepribadian/instruksi khusus bot untuk chat ini
  _Wajib admin grup. /prompt clear untuk menghapus._

*/reset* — hapus memori percakapan bot di chat ini
  _Wajib admin grup._

*/permission* [0-3] — atur izin moderasi (khusus grup)
  _Wajib admin grup._
  0 = tidak bisa kick/delete _(default)_
  1 = boleh delete pesan
  2 = boleh kick member
  3 = boleh delete & kick

*/mode* [auto|prefix|hybrid] — atur mode respons (khusus grup)
  _Wajib owner atau admin grup._
  auto = LLM memutuskan kapan merespons
  prefix = hanya merespons jika ada trigger _(default)_
  hybrid = prefix dulu, fallback ke auto

*/trigger* [tag|reply|name|join|all|none] — toggle trigger di mode prefix
  _Wajib owner atau admin grup._
  tag = bot di-@mention
  reply = seseorang membalas pesan bot
  name = nama bot disebut dalam teks
  join = member baru masuk grup
  all/none = aktifkan/nonaktifkan semua

*/join* <link> — masuk ke grup lewat link undangan

*/dashboard* — lihat statistik penggunaan bot

*/help* — tampilkan pesan ini`;

async function handleHelp({ chatId }) {
  const sock = getSock();
  try {
    await sock.sendMessage(chatId, { text: HELP_TEXT });
  } catch (err) {
    logger.warn({ err, chatId }, 'failed sending /help response');
  }
}

async function handlePrompt({ chatId, chatType, senderIsAdmin, args }) {
  const sock = getSock();
  const isPrivate = chatType === 'private';

  if (!isPrivate && !senderIsAdmin) {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins can use /prompt.' });
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
        await sock.sendMessage(chatId, { text: 'No custom prompt set for this chat. Use /prompt <text> to set one.' });
      } catch (err) { /* ignore */ }
    }
    return;
  }

  if (args.trim().toLowerCase() === '-' || args.trim().toLowerCase() === 'clear' || args.trim().toLowerCase() === 'reset') {
    setPrompt(chatId, null);
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
  const preview = args.length > 200 ? args.slice(0, 197) + '...' : args;
  try {
    await sock.sendMessage(chatId, { text: `Prompt updated:\n${preview}` });
  } catch (err) { /* ignore */ }
}

async function handleReset({ chatId, chatType, senderIsAdmin, contextMsgId }) {
  const sock = getSock();
  const isPrivate = chatType === 'private';

  if (!isPrivate && !senderIsAdmin) {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins can use /reset.' });
    } catch (err) { /* ignore */ }
    return;
  }

  clearSettings(chatId);

  wsClient.send({ type: 'clear_history', chatId });

  try {
    await sock.sendMessage(chatId, { text: 'Bot memory for this chat has been reset.' });
  } catch (err) { /* ignore */ }

  logger.info({ chatId }, 'Memory cleared via /reset');
}

async function handlePermission({ chatId, chatType, senderIsAdmin, botIsAdmin, args }) {
  const sock = getSock();

  if (chatType === 'private') {
    try {
      await sock.sendMessage(chatId, { text: '/permission can only be used in group chats.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (!senderIsAdmin) {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins can use /permission.' });
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

async function handleMode({ chatId, chatType, senderIsAdmin, senderId, args }) {
  const sock = getSock();

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

  if (!isOwnerJid(senderId) && !senderIsAdmin) {
    try {
      await sock.sendMessage(chatId, { text: 'Only the bot owner or group admins can change the mode.' });
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

async function handleTrigger({ chatId, chatType, senderIsAdmin, senderId, args }) {
  const sock = getSock();

  if (!args) {
    const current = getTriggers(chatId);
    if (current.size > 0) {
      const lines = [...current].sort().map((t) => `  - ${t}: ${TRIGGER_DESCRIPTIONS[t] || t}`);
      try {
        await sock.sendMessage(chatId, { text: 'Current triggers:\n' + lines.join('\n') });
      } catch (err) { /* ignore */ }
    } else {
      try {
        await sock.sendMessage(chatId, { text: 'No triggers enabled. Bot won\'t respond in prefix mode.\nUse /trigger all to enable all triggers.' });
      } catch (err) { /* ignore */ }
    }
    return;
  }

  if (!isOwnerJid(senderId) && !senderIsAdmin) {
    try {
      await sock.sendMessage(chatId, { text: 'Only the bot owner or group admins can change triggers.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const cleaned = args.trim().toLowerCase();

  if (cleaned === 'all') {
    setTriggers(chatId, VALID_TRIGGERS);
    try {
      await sock.sendMessage(chatId, { text: 'All triggers enabled: ' + [...VALID_TRIGGERS].sort().join(', ') });
    } catch (err) { /* ignore */ }
    return;
  }

  if (cleaned === 'none') {
    setTriggers(chatId, new Set());
    try {
      await sock.sendMessage(chatId, { text: 'All triggers disabled. Bot won\'t respond in prefix mode.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const requested = new Set(cleaned.split(',').map((t) => t.trim()).filter(Boolean));
  const invalid = [...requested].filter((t) => !VALID_TRIGGERS.has(t));
  if (invalid.length > 0) {
    try {
      await sock.sendMessage(chatId, { text: `Invalid trigger(s): ${invalid.sort().join(', ')}\nValid: ${[...VALID_TRIGGERS].sort().join(', ')}` });
    } catch (err) { /* ignore */ }
    return;
  }

  const current = getTriggers(chatId);
  const toggledOn = new Set([...requested].filter((t) => !current.has(t)));
  const toggledOff = new Set([...requested].filter((t) => current.has(t)));
  const newTriggers = new Set([...current, ...toggledOn]);
  for (const t of toggledOff) newTriggers.delete(t);
  setTriggers(chatId, newTriggers);

  const statusLines = [...requested].sort().map((t) => `  - ${t}: ${toggledOn.has(t) ? 'enabled' : 'disabled'}`);
  const activeStr = newTriggers.size > 0 ? [...newTriggers].sort().join(', ') : 'none';
  try {
    await sock.sendMessage(chatId, { text: statusLines.join('\n') + `\nActive triggers: ${activeStr}` });
  } catch (err) { /* ignore */ }
}

async function handleDashboard({ chatId }) {
  const sock = getSock();
  const now = new Date();
  const dailyKey = now.toISOString().slice(0, 10);
  const weekKey = getWeekKey(now);
  const monthKey = now.toISOString().slice(0, 7);

  const daily = getStats(chatId, 'daily', dailyKey);
  const weekly = getStats(chatId, 'weekly', weekKey);
  const monthly = getStats(chatId, 'monthly', monthKey);
  const topUsers = getTopUsers(chatId, 'monthly', monthKey, 5);

  const lines = ['*Dashboard Stats*'];
  lines.push('');
  lines.push(`*Daily (${dailyKey})*`);
  lines.push(`  Messages processed: ${daily.messages_processed || 0}`);
  lines.push(`  Responses sent: ${daily.responses_sent || 0}`);
  lines.push(`  Bot tags: ${daily.bot_tags || 0}`);
  lines.push(`  Stickers sent: ${daily.stickers_sent || 0}`);
  lines.push('');
  lines.push(`*Weekly (${weekKey})*`);
  lines.push(`  Messages processed: ${weekly.messages_processed || 0}`);
  lines.push(`  Responses sent: ${weekly.responses_sent || 0}`);
  lines.push('');
  lines.push(`*Monthly (${monthKey})*`);
  lines.push(`  Messages processed: ${monthly.messages_processed || 0}`);
  lines.push(`  Responses sent: ${monthly.responses_sent || 0}`);
  lines.push(`  LLM1 calls: ${monthly.llm1_calls || 0}`);
  lines.push(`  LLM2 calls: ${monthly.llm2_calls || 0}`);

  if (topUsers.length > 0) {
    lines.push('');
    lines.push('*Top Users (Monthly)*');
    for (const u of topUsers) {
      lines.push(`  ${u.senderName || u.senderRef}: ${u.invokeCount}`);
    }
  }

  try {
    await sock.sendMessage(chatId, { text: lines.join('\n') });
  } catch (err) {
    logger.warn({ err, chatId }, 'failed sending /dashboard response');
  }
}

function getWeekKey(date) {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() - d.getDay() + 1);
  return d.toISOString().slice(0, 10);
}

async function handleCommandListener(msg, context) {
  const { slashCommand, chatId, chatType, senderIsAdmin, senderId, botIsAdmin, contextMsgId } = context;

  if (!slashCommand) return false;

  const { command, args } = slashCommand;

  switch (command) {
    case 'help':
      await handleHelp({ chatId });
      return true;

    case 'prompt':
      await handlePrompt({ chatId, chatType, senderIsAdmin, args });
      return true;

    case 'reset':
      await handleReset({ chatId, chatType, senderIsAdmin, contextMsgId });
      return true;

    case 'permission':
      await handlePermission({ chatId, chatType, senderIsAdmin, botIsAdmin, args });
      return true;

    case 'mode':
      await handleMode({ chatId, chatType, senderIsAdmin, senderId, args });
      return true;

    case 'trigger':
      await handleTrigger({ chatId, chatType, senderIsAdmin, senderId, args });
      return true;

    case 'dashboard':
      await handleDashboard({ chatId });
      return true;

    case 'broadcast':
      await handleBroadcastCommand({
        chatId,
        senderId,
        text: args,
        quotedMessageId: null,
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
      return false;

    default:
      return false;
  }
}

export {
  handleCommandListener,
  handleHelp,
  handlePrompt,
  handleReset,
  handlePermission,
  handleMode,
  handleTrigger,
  handleDashboard,
};
