import logger from '../logger.js';
import { isOwnerJid } from '../participants.js';
import { messageCache } from '../caches.js';
import { getSock } from './connection.js';
import { sendNativeFlow, sendCarousel } from './interactive/index.js';

// ---------------------------------------------------------------------------
// Slash command parsing
// ---------------------------------------------------------------------------

const SLASH_CMD_RE = /^\/(broadcast|prompt|reset|permission|info|mode|trigger|dashboard|help|debug)\b\s*([\s\S]*)/i;

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

// ---------------------------------------------------------------------------
// /debug command
// ---------------------------------------------------------------------------

const DEBUG_TYPES = ['buttons', 'menu', 'carousel', 'carousel-img', 'all'];

async function sendDebugButtons(chatId) {
  const sock = getSock();
  // quick_reply × 3
  await sendNativeFlow(sock, chatId, '[DEBUG] quick_reply buttons', [
    { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Option A', id: 'debug_qr_a' }) },
    { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Option B', id: 'debug_qr_b' }) },
    { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Option C', id: 'debug_qr_c' }) },
  ], { footer: 'Tap any button to test' });

  // cta_url
  await sendNativeFlow(sock, chatId, '[DEBUG] cta_url button', [
    {
      name: 'cta_url',
      buttonParamsJson: JSON.stringify({
        display_text: 'Open Link',
        url: 'https://github.com/chomosuke9/wazzapagents',
        merchant_url: 'https://github.com/chomosuke9/wazzapagents',
      }),
    },
  ], { footer: 'Opens a URL' });

  // cta_copy
  await sendNativeFlow(sock, chatId, '[DEBUG] cta_copy button', [
    {
      name: 'cta_copy',
      buttonParamsJson: JSON.stringify({ display_text: 'Copy Code', id: 'debug_copy', copy_code: 'DEBUG-CODE-123' }),
    },
  ], { footer: 'Tap to copy code to clipboard' });

  // cta_call
  await sendNativeFlow(sock, chatId, '[DEBUG] cta_call button', [
    {
      name: 'cta_call',
      buttonParamsJson: JSON.stringify({ display_text: 'Call Now', id: 'debug_call', phone_number: '+621234567890' }),
    },
  ], { footer: 'Tap to call' });
}

async function sendDebugMenu(chatId) {
  const sock = getSock();
  await sendNativeFlow(sock, chatId, '[DEBUG] single_select (menu/dropdown)', [
    {
      name: 'single_select',
      buttonParamsJson: JSON.stringify({
        title: 'Pilih opsi',
        sections: [
          {
            title: 'Kategori 1',
            rows: [
              { title: 'Item A', description: 'Deskripsi item A', id: 'debug_menu_a' },
              { title: 'Item B', description: 'Deskripsi item B', id: 'debug_menu_b' },
            ],
          },
          {
            title: 'Kategori 2',
            rows: [
              { title: 'Item C', id: 'debug_menu_c' },
              { title: 'Item D', id: 'debug_menu_d' },
            ],
          },
        ],
      }),
    },
  ], { footer: 'Tap to open dropdown menu' });
}

async function sendDebugCarousel(chatId) {
  const sock = getSock();
  await sendCarousel(sock, chatId, [
    {
      body: 'Kartu 1 — quick_reply buttons',
      footer: 'Footer kartu 1',
      buttons: [
        { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Pilih Ini', id: 'debug_c1_qr' }) },
        {
          name: 'cta_url',
          buttonParamsJson: JSON.stringify({
            display_text: 'Buka Link',
            url: 'https://github.com/chomosuke9/wazzapagents',
            merchant_url: 'https://github.com/chomosuke9/wazzapagents',
          }),
        },
      ],
    },
    {
      body: 'Kartu 2 — cta_copy & cta_call',
      footer: 'Footer kartu 2',
      buttons: [
        { name: 'cta_copy', buttonParamsJson: JSON.stringify({ display_text: 'Salin Kode', id: 'debug_c2_copy', copy_code: 'CAROUSEL-456' }) },
        { name: 'cta_call', buttonParamsJson: JSON.stringify({ display_text: 'Hubungi', id: 'debug_c2_call', phone_number: '+621234567890' }) },
      ],
    },
    {
      body: 'Kartu 3 — single_select',
      footer: 'Footer kartu 3',
      buttons: [
        {
          name: 'single_select',
          buttonParamsJson: JSON.stringify({
            title: 'Pilih dari menu',
            sections: [
              {
                title: 'Pilihan',
                rows: [
                  { title: 'Opsi X', id: 'debug_c3_x' },
                  { title: 'Opsi Y', id: 'debug_c3_y' },
                ],
              },
            ],
          }),
        },
      ],
    },
  ], { text: '[DEBUG] carousel message' });
}

// Default fallback image — picsum.photos is a standard dev placeholder service
const DEBUG_IMG_DEFAULT = 'https://picsum.photos/seed/wazzap/600/400';
const DEBUG_IMG_MIME = 'image/jpeg';

async function sendDebugCarouselImg(chatId, imageUrl) {
  const sock = getSock();
  const url = imageUrl || DEBUG_IMG_DEFAULT;
  await sendCarousel(sock, chatId, [
    {
      image: { url },
      body: 'Kartu 1 — header image + quick_reply',
      footer: 'Footer kartu 1',
      buttons: [
        { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Pilih Ini', id: 'debug_ci1_qr' }) },
        {
          name: 'cta_url',
          buttonParamsJson: JSON.stringify({
            display_text: 'Buka Link',
            url: 'https://github.com/chomosuke9/wazzapagents',
            merchant_url: 'https://github.com/chomosuke9/wazzapagents',
          }),
        },
      ],
    },
    {
      image: { url },
      body: 'Kartu 2 — header image + cta_copy & cta_call',
      footer: 'Footer kartu 2',
      buttons: [
        { name: 'cta_copy', buttonParamsJson: JSON.stringify({ display_text: 'Salin Kode', id: 'debug_ci2_copy', copy_code: 'IMG-CAROUSEL-789' }) },
        { name: 'cta_call', buttonParamsJson: JSON.stringify({ display_text: 'Hubungi', id: 'debug_ci2_call', phone_number: '+621234567890' }) },
      ],
    },
    {
      // No image — compare rendering with vs without image
      body: 'Kartu 3 — tanpa header image (baseline)',
      footer: 'Footer kartu 3',
      buttons: [
        { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Baseline', id: 'debug_ci3_qr' }) },
      ],
    },
  ], { text: `[DEBUG] carousel + image header (${url})` });
}

async function handleDebugCommand({ chatId, senderId, args }) {
  const sock = getSock();
  if (!isOwnerJid(senderId)) {
    logger.info({ senderId, chatId }, '/debug rejected: not owner');
    try {
      await sock.sendMessage(chatId, { text: 'Only bot owners can use /debug.' });
    } catch (err) {
      logger.warn({ err }, 'failed sending debug rejection');
    }
    return;
  }

  const [subTypeRaw = '', ...restParts] = (args || '').trim().split(/\s+/);
  const subType = subTypeRaw.toLowerCase();
  const extraArg = restParts.join(' ').trim();

  if (!subType || !DEBUG_TYPES.includes(subType)) {
    try {
      await sock.sendMessage(chatId, {
        text: [
          'Usage: /debug <type>',
          '',
          `Types: ${DEBUG_TYPES.join(', ')}`,
          '',
          '- buttons      → quick_reply, cta_url, cta_copy, cta_call',
          '- menu         → single_select dropdown',
          '- carousel     → swipeable cards (tanpa header image)',
          '- carousel-img → swipeable cards dengan header image',
          '                 Opsional: /debug carousel-img <url>',
          '- all          → buttons + menu + carousel + carousel-img',
        ].join('\n'),
      });
    } catch (err) {
      logger.warn({ err }, 'failed sending debug usage');
    }
    return;
  }

  const send = async (fn, label, ...fnArgs) => {
    try {
      await fn(chatId, ...fnArgs);
      logger.info({ chatId, label }, 'debug interactive message sent');
    } catch (err) {
      logger.warn({ err, label }, 'debug send failed');
      try {
        await sock.sendMessage(chatId, { text: `❌ Gagal mengirim ${label}: ${err?.message || err}` });
      } catch (e) { /* ignore */ }
    }
  };

  if (subType === 'buttons' || subType === 'all') await send(sendDebugButtons, 'buttons');
  if (subType === 'menu' || subType === 'all') await send(sendDebugMenu, 'menu');
  if (subType === 'carousel' || subType === 'all') await send(sendDebugCarousel, 'carousel');
  if (subType === 'carousel-img' || subType === 'all') await send(sendDebugCarouselImg, 'carousel-img', extraArg || null);
}

export {
  parseSlashCommand,
  handleBroadcastCommand,
  truncateText,
  handleInfoCommand,
  handleDebugCommand,
};
