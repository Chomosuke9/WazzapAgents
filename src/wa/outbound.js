import path from 'path';
import logger from '../logger.js';
import {
  normalizeJid,
  normalizeContextMsgId,
  nextContextMsgId,
  rememberSenderRef,
  rememberMessage,
  isPhoneJid,
  resolveMentionTargetBySenderRef,
  mentionHandleForJid,
  resolveQuotedMessage,
} from '../identifiers.js';
import { getGroupContext } from '../groupContext.js';
import { resolveAllowedAttachmentPath } from '../mediaHandler.js';
import { getSock } from './connection.js';
import { escapeRegex } from './utils.js';
import { actionError } from './actions.js';

async function renderOutboundMentions(chatId, rawText, groupContext = null) {
  if (typeof rawText !== 'string') {
    return { text: rawText, mentions: [], groupContext };
  }
  // Match @Name (senderRef) pattern — name can contain spaces, non-greedy to handle multiple mentions
  const tokens = Array.from(rawText.matchAll(/@(.+?)\s*\(([^)\r\n]+)\)/g));
  if (tokens.length === 0) {
    return { text: rawText, mentions: [], groupContext };
  }

  let resolvedGroup = groupContext;
  let retried = false;
  let cursor = 0;
  let rendered = '';
  const mentionSet = new Set();

  for (const token of tokens) {
    const fullToken = token[0];
    const rawName = typeof token[1] === 'string' ? token[1].trim() : '';
    const rawValue = typeof token[2] === 'string' ? token[2].trim() : '';
    const normalizedValue = rawValue.toLowerCase();
    const index = Number.isInteger(token.index) ? token.index : -1;
    if (index < 0) continue;

    rendered += rawText.slice(cursor, index);
    let replacement = rawName ? `@${rawName}` : '@';

    if (normalizedValue === 'everyone') {
      if (chatId?.endsWith('@g.us')) {
        let participants = Array.isArray(resolvedGroup?.participants) ? resolvedGroup.participants : [];
        if (participants.length === 0) {
          resolvedGroup = await getGroupContext(chatId, { forceRefresh: true });
          participants = Array.isArray(resolvedGroup?.participants) ? resolvedGroup.participants : [];
        }
        for (const participantJid of participants) {
          const normalizedParticipant = normalizeJid(participantJid) || participantJid;
          if (!normalizedParticipant) continue;
          mentionSet.add(normalizedParticipant);
        }
      }
      replacement = '@everyone';
    } else if (normalizedValue === 'bot') {
      // Bot mention — render as display name, no JID resolution needed
      replacement = rawName ? `@${rawName}` : '@bot';
    } else if (normalizedValue) {
      let participantJid = resolveMentionTargetBySenderRef(chatId, normalizedValue);
      if (!participantJid && !retried && chatId?.endsWith('@g.us')) {
        logger.debug({ chatId, senderRef: normalizedValue }, 'senderRef not found — force-refreshing group metadata');
        resolvedGroup = await getGroupContext(chatId, { forceRefresh: true });
        retried = true;
        participantJid = resolveMentionTargetBySenderRef(chatId, normalizedValue);
      }
      if (participantJid) {
        const normalizedParticipant = normalizeJid(participantJid) || participantJid;
        mentionSet.add(normalizedParticipant);
        replacement = mentionHandleForJid(normalizedParticipant) || replacement;
      } else {
        logger.warn({ chatId, senderRef: normalizedValue }, 'outbound mention resolution failed — token will render as plain text');
      }
    }

    rendered += replacement;
    cursor = index + fullToken.length;
  }

  rendered += rawText.slice(cursor);
  const mentionsArray = Array.from(mentionSet);
  for (const jid of mentionsArray) {
    if (!isPhoneJid(jid)) {
      logger.warn({ chatId, jid }, 'outbound mention contains non-phone JID — may not render as clickable');
    }
  }
  return {
    text: rendered,
    mentions: mentionsArray,
    groupContext: resolvedGroup,
  };
}

async function sendOutgoing({ chatId, text, attachments = [], replyTo }) {
  const sock = getSock();
  if (!sock) throw actionError('send_failed', 'WhatsApp socket not ready');
  if (!chatId) throw actionError('invalid_target', 'chatId is required');
  if (attachments != null && !Array.isArray(attachments)) {
    throw actionError('invalid_target', 'attachments must be an array');
  }

  const quoted = replyTo ? resolveQuotedMessage(chatId, replyTo) : null;
  if (replyTo && !quoted) {
    throw actionError('not_found', 'reply target not found');
  }

  const isGroup = chatId.endsWith('@g.us');
  let group = isGroup ? await getGroupContext(chatId) : null;
  const botSenderId = normalizeJid(sock.user?.id) || 'bot@wazzap.local';
  const botSenderRef = rememberSenderRef(chatId, botSenderId, botSenderId) || 'unknown';
  const normalizedText = typeof text === 'string' ? text.trim() : '';
  const normalizedAttachments = Array.isArray(attachments) ? attachments : [];
  if (!normalizedText && normalizedAttachments.length === 0) {
    throw actionError('invalid_target', 'send_message requires non-empty text or at least one attachment');
  }
  const sent = [];

  // send attachments first (with caption if provided)
  for (const att of normalizedAttachments) {
    const kindToken = typeof att?.kind === 'string' ? att.kind : (typeof att?.type === 'string' ? att.type : 'document');
    const kind = kindToken.trim().toLowerCase() || 'document';
    const filePath = await resolveAllowedAttachmentPath(att?.path, actionError);
    const content = {};
    if (kind === 'image') content.image = { url: filePath };
    else if (kind === 'video') content.video = { url: filePath };
    else if (kind === 'audio') content.audio = { url: filePath, ptt: false };
    else if (kind === 'sticker') content.sticker = { url: filePath };
    else content.document = { url: filePath, fileName: att.fileName || path.basename(filePath) };

    if (att.caption) {
      const renderedCaption = await renderOutboundMentions(chatId, String(att.caption), group);
      content.caption = renderedCaption.text;
      if (renderedCaption.mentions.length > 0) {
        content.mentions = renderedCaption.mentions;
      }
      group = renderedCaption.groupContext || group;
    }

    const sentMsg = await sock.sendMessage(chatId, content, quoted ? { quoted } : {});
    const contextMsgId = nextContextMsgId(chatId);
    rememberMessage(sentMsg, {
      chatId,
      contextMsgId,
      senderId: botSenderId,
      senderRef: botSenderRef,
      senderIsAdmin: Boolean(group?.botIsAdmin),
      fromMe: true,
      timestampMs: Date.now(),
    });
    sent.push({
      kind,
      contextMsgId,
      messageId: sentMsg?.key?.id || null,
    });
  }

  if (normalizedText) {
    const renderedText = await renderOutboundMentions(chatId, normalizedText, group);
    group = renderedText.groupContext || group;
    const textPayload = { text: renderedText.text };
    if (renderedText.mentions.length > 0) {
      textPayload.mentions = renderedText.mentions;
    }
    const sentMsg = await sock.sendMessage(chatId, textPayload, quoted ? { quoted } : {});
    const contextMsgId = nextContextMsgId(chatId);
    rememberMessage(sentMsg, {
      chatId,
      contextMsgId,
      senderId: botSenderId,
      senderRef: botSenderRef,
      senderIsAdmin: Boolean(group?.botIsAdmin),
      fromMe: true,
      timestampMs: Date.now(),
    });
    sent.push({
      kind: 'text',
      contextMsgId,
      messageId: sentMsg?.key?.id || null,
    });
  }
  if (sent.length === 0) {
    throw actionError('invalid_target', 'send_message produced no deliverable content');
  }

  return {
    sent,
    replyTo: normalizeContextMsgId(replyTo),
  };
}

export {
  renderOutboundMentions,
  sendOutgoing,
};
