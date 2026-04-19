import logger from '../logger.js';
import { isOwnerJid, roleFlagsForJid } from '../participants.js';
import { getSock } from './connection.js';
import { sendRichMessage, sendNativeFlow } from './interactive/index.js';
import {
  getPrompt,
  setPrompt,
  getPermission,
  setPermission,
  getMode,
  setMode,
  getTriggers,
  setTriggers,
  getStats,
  getTopUsers,
  getLlm2Model,
  setLlm2Model,
  getAllActiveModels,
  getAllModels,
  getDefaultLlm2Model,
  addModel,
  updateModel,
  deleteModel,
  VALID_MODES,
  VALID_TRIGGERS,
} from '../db.js';
import config from '../config.js';
import wsClient from '../wsClient.js';
import {
  handleBroadcastCommand,
  handleInfoCommand,
  handleDebugCommand,
  handleJoinCommand,
} from './commands.js';
import { sendOutgoing } from './outbound.js';
import { createStickerFile } from './stickerTool.js';
import { unwrapMessage } from '../messageParser.js';
import { downloadMediaToFile, mapMediaKind } from '../mediaHandler.js';
import { withTimeout } from './utils.js';

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

const HELP_TEXT = `*WazzapAgents - Daftar Perintah*

*Umum (Semua Orang)*
• */help* — Tampilkan pesan bantuan ini
• */info* — Informasi profil, peran, dan chat
• */dashboard* — Statistik penggunaan bot
• */join* [link] — Masuk grup via link

*Pengaturan & Moderasi (Admin/Owner)*
• */setting* — Menu pengaturan interaktif
• */prompt* [teks] — Atur kepribadian bot
• */reset* — Hapus memori percakapan
• */model* — Pilih model AI
• */sticker* — Buat stiker (balas gambar)
• */mode* [auto|prefix|hybrid] — Mode respon
• */trigger* [opsi] — Atur pemicu respon

_Ketik perintah tanpa argumen untuk melihat status saat ini._`;

async function handleHelp({ chatId }) {
  const sock = getSock();
  try {
    await sock.sendMessage(chatId, { text: HELP_TEXT });
  } catch (err) {
    logger.warn({ err, chatId }, 'failed sending /help response');
  }
}

async function handlePrompt({ chatId, chatType, senderIsAdmin, senderIsOwner, args }) {
  const sock = getSock();
  const isPrivate = chatType === 'private';

  if (isPrivate || senderIsOwner || senderIsAdmin) {
    // proceed
  } else {
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

async function handleSticker({ chatId, chatType, senderIsAdmin, senderIsOwner, args, msg }) {
  const isPrivate = chatType === 'private';
  const canUse = isPrivate ? senderIsOwner : senderIsAdmin;
  if (!canUse) {
    try {
      await getSock().sendMessage(chatId, { text: 'Only group admins or bot owner can use /sticker.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const [upperText, lowerText] = parseStickerArgs(args);

  const { contentType, message: innerMessage } = unwrapMessage(msg.message) || {};
  let mediaPath = null;

  if (contentType === 'imageMessage') {
    mediaPath = await downloadMediaContent(innerMessage[contentType], contentType, msg.key.id);
  } else if (contentType === 'videoMessage') {
    try {
      await getSock().sendMessage(chatId, { text: '/sticker saat ini hanya mendukung gambar (video belum didukung).' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (!mediaPath && innerMessage?.extendedTextMessage?.contextInfo) {
    const ctx = innerMessage.extendedTextMessage.contextInfo;
    if (ctx.quotedMessage) {
      const { contentType: qType, message: qMsg } = unwrapMessage(ctx.quotedMessage) || {};
      const qContent = qType ? qMsg?.[qType] : null;
      if (qType === 'imageMessage') {
        mediaPath = await downloadMediaContent(qContent, qType, ctx.stanzaId);
      } else if (qType === 'videoMessage') {
        try {
          await getSock().sendMessage(chatId, { text: '/sticker saat ini hanya mendukung gambar (video belum didukung).' });
        } catch (err) { /* ignore */ }
        return;
      }
    }
  }

  if (!mediaPath) {
    try {
      await getSock().sendMessage(chatId, { text: 'Send an image with /sticker caption, or reply to an image.' });
    } catch (err) { /* ignore */ }
    return;
  }

  try {
    const stickerPath = await createStickerFile(mediaPath, upperText, lowerText);
    await sendOutgoing({
      chatId,
      attachments: [{ kind: 'sticker', path: stickerPath }],
      replyTo: msg.key.id,
    });
    logger.info({ chatId }, 'Sticker created and sent');
  } catch (err) {
    logger.error({ err, chatId }, 'failed to create sticker');
    try {
      await getSock().sendMessage(chatId, { text: `Failed to create sticker: ${err.message}` });
    } catch (e) { /* ignore */ }
  }
}

function parseStickerArgs(args) {
  if (!args || !args.trim()) return [null, null];
  if (args.includes('#')) {
    const [upper, lower] = args.split('#');
    return [upper.trim() || null, lower.trim() || null];
  }
  return [args.trim() || null, null];
}

async function downloadMediaContent(content, contentType, messageId) {
  const mediaKind = mapMediaKind(contentType);
  if (!mediaKind || mediaKind !== 'image') return null;

  try {
    const ext = mediaKind === 'video' ? 'mp4' : 'jpg';
    const filename = `${messageId}_${mediaKind}.${ext}`;
    const filepath = `${config.mediaDir}/${filename}`;
    await downloadMediaToFile(content, mediaKind, filepath, withTimeout);
    return filepath;
  } catch (err) {
    logger.warn({ err, messageId, contentType }, 'failed to download media for sticker');
    return null;
  }
}

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

async function handleTrigger({ chatId, chatType, senderIsAdmin, senderIsOwner, senderId, args }) {
  const sock = getSock();

  if (chatType === 'private') {
    try {
      await sock.sendMessage(chatId, { text: '/trigger can only be used in group chats.' });
    } catch (err) { /* ignore */ }
    return;
  }

  if (!senderIsOwner && !senderIsAdmin) {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins can change triggers.' });
    } catch (err) { /* ignore */ }
    return;
  }

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

async function handleModel({ chatId, chatType, senderIsAdmin, senderIsOwner, args }) {
  const sock = getSock();
  const isPrivate = chatType === 'private';
  const canUse = isPrivate || senderIsOwner || senderIsAdmin;

  if (!canUse) {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins or bot owner can change the model.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const models = getAllActiveModels();
  if (models.length === 0) {
    try {
      await sock.sendMessage(chatId, { text: 'No models available. Ask the bot owner to add models using /modelcfg.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const currentModelId = getLlm2Model(chatId);
  const defaultModel = getDefaultLlm2Model();
  const activeModelId = currentModelId || defaultModel?.modelId || null;
  const activeModel = models.find((m) => m.modelId === activeModelId);

  const sections = models.map((m) => ({
    title: m.displayName,
    rows: [{
      title: m.displayName + (m.modelId === activeModelId ? ' ✓' : ''),
      description: m.description || '',
      id: `model_select:${m.modelId}`,
    }],
  }));

  try {
    await sendNativeFlow(sock, chatId, 'Pilih Model LLM', [
      {
        name: 'single_select',
        buttonParamsJson: JSON.stringify({
          title: 'Pilih Model',
          sections,
        }),
      },
    ], { footer: 'Model saat ini: ' + (activeModel?.displayName || 'default') });
  } catch (err) {
    logger.warn({ err, chatId }, 'failed sending /model interactive');
    try {
      await sock.sendMessage(chatId, { text: 'Failed to show model list. Please try again.' });
    } catch (e) { /* ignore */ }
  }
}

async function handleModelcfg({ chatId, senderId, senderIsOwner, args }) {
  const sock = getSock();
  if (!senderIsOwner) {
    try {
      await sock.sendMessage(chatId, { text: 'Only bot owner can use /modelcfg.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const parts = (args || '').trim().split(/\s+/);
  const subcommand = parts[0]?.toLowerCase() || '';
  const subArgs = parts.slice(1);

  if (!subcommand) {
    const modelRows = getAllActiveModels().map((m) => ({
      id: `modelcfg_default:${m.modelId}`,
      title: m.displayName,
      description: m.description || '',
    }));
    
    const allModelRows = getAllModels().map((m) => ({
      id: `modelcfg_remove:${m.modelId}`,
      title: m.displayName + (m.isActive ? '' : ' (inactive)'),
      description: m.description || `ID: ${m.modelId}`,
    }));

    const editModelRows = getAllModels().map((m) => ({
      id: `/modelcfg edit ${m.modelId}`,
      title: m.displayName + (m.isActive ? '' : ' (inactive)'),
      description: m.description || `ID: ${m.modelId}`,
    }));

    const buttons = [
      {
        name: 'single_select',
        buttonParamsJson: JSON.stringify({
          title: 'Default Model',
          sections: [{
            title: 'Select Default',
            rows: modelRows,
          }],
        }),
      },
      {
        name: 'single_select',
        buttonParamsJson: JSON.stringify({
          title: 'Edit Model',
          sections: [{
            title: 'Select Model to Edit',
            rows: editModelRows,
          }],
        }),
      },
      {
        name: 'single_select',
        buttonParamsJson: JSON.stringify({
          title: 'Remove Model',
          sections: [{
            title: 'Select Model to Remove',
            rows: allModelRows,
          }],
        }),
      },
    ];

    try {
      await sendNativeFlow(sock, chatId, 'Model Configuration', buttons, { footer: 'Bot Owner Only' });
    } catch (err) {
      logger.warn({ err, chatId }, 'failed sending /modelcfg menu');
      try {
        await sock.sendMessage(chatId, { text: 'Failed to show modelcfg menu.' });
      } catch (e) { /* ignore */ }
    }
    return;
  }

  if (subcommand === 'remove_menu') {
    const models = getAllModels();
    if (models.length === 0) {
      try {
        await sock.sendMessage(chatId, { text: 'No models to remove.' });
      } catch (err) { /* ignore */ }
      return;
    }
    const sections = [
      {
        title: 'Select Model to Remove',
        rows: models.map((m) => ({
          title: m.displayName + (m.isActive ? '' : ' (inactive)'),
          description: m.description || `ID: ${m.modelId}`,
          id: `modelcfg_remove:${m.modelId}`,
        })),
      },
    ];
    try {
      await sendNativeFlow(sock, chatId, '⚠️ Remove Model', [
        {
          name: 'single_select',
          buttonParamsJson: JSON.stringify({
            title: 'Hapus Model',
            sections,
          }),
        },
      ], { footer: 'Pilih model untuk dihapus' });
    } catch (err) {
      logger.warn({ err, chatId }, 'failed sending /modelcfg remove menu');
      try {
        await sock.sendMessage(chatId, { text: 'Failed to show remove menu.' });
      } catch (e) { /* ignore */ }
    }
    return;
  }

  switch (subcommand) {
    case 'list': {
      const models = getAllModels();
      if (models.length === 0) {
        try {
          await sock.sendMessage(chatId, { text: 'No models configured. Use /modelcfg add <model_id> <display_name> [description]' });
        } catch (err) { /* ignore */ }
        return;
      }
      const lines = ['*Daftar Model:*'];
      const defaultModel = getDefaultLlm2Model();
      for (const m of models) {
        const isDefault = defaultModel?.modelId === m.modelId;
        const status = m.isActive ? '✓' : '✗';
        lines.push(`${status} ${m.displayName} (${m.modelId})${isDefault ? ' [DEFAULT]' : ''}`);
        if (m.description) lines.push(`   ${m.description}`);
      }
      try {
        await sock.sendMessage(chatId, { text: lines.join('\n') });
      } catch (err) { /* ignore */ }
      break;
    }

    case 'add': {
      if (subArgs.length < 2) {
        try {
          await sock.sendMessage(chatId, { text: 'Usage: /modelcfg add <model_id> <display_name> [description]' });
        } catch (err) { /* ignore */ }
        return;
      }
      const [modelId, displayName, ...descParts] = subArgs;
      const description = descParts.join(' ');
      const success = addModel(modelId, displayName, description);
      if (success) {
        wsClient.sendReliable({ type: 'invalidate_default_model' });
      }
      try {
        await sock.sendMessage(chatId, { text: success ? `Model "${displayName}" added.` : `Model "${modelId}" already exists.` });
      } catch (err) { /* ignore */ }
      break;
    }

    case 'edit': {
      if (subArgs.length < 1) {
        try {
          await sock.sendMessage(chatId, { text: 'Usage: /modelcfg edit <model_id> [name=<name>] [desc=<desc>] [active=0|1] [order=<n>]' });
        } catch (err) { /* ignore */ }
        return;
      }
      const [modelId, ...editParts] = subArgs;
      const updates = {};
      for (const part of editParts) {
        const match = part.match(/^(name|desc|active|order)=(.+)$/);
        if (match) {
          const [, key, value] = match;
          if (key === 'name') updates.displayName = value;
          else if (key === 'desc') updates.description = value;
          else if (key === 'active') updates.isActive = value === '1' || value === 'true';
          else if (key === 'order') updates.sortOrder = parseInt(value, 10);
        }
      }
      const success = updateModel(modelId, updates);
      if (success) {
        wsClient.sendReliable({ type: 'invalidate_default_model' });
      }
      try {
        await sock.sendMessage(chatId, { text: success ? `Model "${modelId}" updated.` : `Model "${modelId}" not found.` });
      } catch (err) { /* ignore */ }
      break;
    }

    case 'remove':
    case 'delete': {
      if (subArgs.length < 1) {
        try {
          await sock.sendMessage(chatId, { text: 'Usage: /modelcfg remove <model_id>' });
        } catch (err) { /* ignore */ }
        return;
      }
      const [modelId] = subArgs;
      const success = deleteModel(modelId);
      if (success) {
        wsClient.sendReliable({ type: 'invalidate_default_model' });
      }
      try {
        await sock.sendMessage(chatId, { text: success ? `Model "${modelId}" deleted.` : `Model "${modelId}" not found.` });
      } catch (err) { /* ignore */ }
      break;
    }

    default:
      try {
        await sock.sendMessage(chatId, { text: 'Unknown subcommand. Use: list, add, edit, remove' });
      } catch (err) { /* ignore */ }
  }
}

async function handleSettings({ chatId, chatType, senderId, senderIsAdmin, senderIsOwner, args }) {
  const sock = getSock();
  const isPrivate = chatType === 'private';
  const canUse = isPrivate || senderIsOwner || senderIsAdmin;

  if (!canUse) {
    try {
      await sock.sendMessage(chatId, { text: 'Only group admins can access settings.' });
    } catch (err) { /* ignore */ }
    return;
  }

  const currentModelId = getLlm2Model(chatId);
  const defaultModel = getDefaultLlm2Model();
  const activeModelId = currentModelId || defaultModel?.modelId;
  const activeModelName = (activeModelId ? getAllModels().find((m) => m.modelId === activeModelId)?.displayName : null) || defaultModel?.displayName || 'default';

  const currentPermission = getPermission(chatId);
  const currentMode = getMode(chatId);

  const permissionLabels = ['Forbidden', 'Delete only', 'Delete & mute', 'All moderation'];
  const permissionLabel = permissionLabels[currentPermission] || String(currentPermission);

  const buttons = [
    {
      name: 'single_select',
      buttonParamsJson: JSON.stringify({
        title: 'Change Mode',
        sections: [{
          title: 'Select Mode',
          rows: [
            { id: '/mode auto', title: 'Auto', description: 'LLM decides when to respond' },
            { id: '/mode prefix', title: 'Prefix', description: 'Only responds when triggered' },
            { id: '/mode hybrid', title: 'Hybrid', description: 'Prefix first, fallback to auto' },
          ],
        }],
      }),
    },
    {
      name: 'single_select',
      buttonParamsJson: JSON.stringify({
        title: 'Change Model',
        sections: [{
          title: 'Select Model',
          rows: getAllActiveModels().map((m) => ({
            id: `model_select:${m.modelId}`,
            title: m.displayName,
            description: m.description || '',
          })),
        }],
      }),
    },
    {
      name: 'single_select',
      buttonParamsJson: JSON.stringify({
        title: 'Set Permission',
        sections: [{
          title: 'Permission Level',
          rows: [
            { id: '/permission 0', title: 'Level 0 - Forbidden', description: 'No moderation' },
            { id: '/permission 1', title: 'Level 1 - Delete', description: 'Can delete' },
            { id: '/permission 2', title: 'Level 2 - Mute', description: 'Delete & mute' },
            { id: '/permission 3', title: 'Level 3 - All', description: 'Delete, mute & kick' },
          ],
        }],
      }),
    },
    {
      name: 'single_select',
      buttonParamsJson: JSON.stringify({
        title: 'Misc',
        sections: [{
          title: 'Misc Options',
          rows: [
            { id: '/prompt', title: 'Get Prompt', description: 'View current prompt' },
            { id: '/reset', title: 'Reset Chat', description: 'Clear bot memory for this chat' },
          ],
        }],
      }),
    },
  ];

  try {
    await sendNativeFlow(sock, chatId, `Chat Settings\n\nCurrent:\n- Mode: ${currentMode}\n- Model: ${activeModelName}\n- Permission: Level ${currentPermission} (${permissionLabel})`, buttons, { footer: 'Click a button' });
  } catch (err) {
    logger.warn({ err, chatId }, 'failed sending /settings interactive');
    try {
      await sock.sendMessage(chatId, { text: 'Failed to show settings menu.' });
    } catch (e) { /* ignore */ }
  }
}

function getWeekKey(date) {
  const d = new Date(date);
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() - d.getDay() + 1);
  return d.toISOString().slice(0, 10);
}

async function handleCommandListener(msg, context) {
  const { slashCommand, chatId, chatType, senderIsAdmin, senderId, botIsAdmin, senderIsOwner, contextMsgId } = context;

  if (!slashCommand) return false;

  const { command, args } = slashCommand;

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
