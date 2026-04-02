/**
 * sendInteractive.js — NativeFlow-based interactive messages.
 * (quick reply, URL, copy, call, list, combined, raw native flow)
 *
 * interactiveMessage content must be sent via generateWAMessageFromContent +
 * sock.relayMessage — sock.sendMessage does not support interactiveMessage
 * because Baileys tries to process it through prepareWAMessageMedia and fails.
 * sendList uses listMessage which works fine with sock.sendMessage directly.
 */
import { proto, generateWAMessageFromContent } from 'baileys';
import logger from '../../logger.js';

/**
 * Internal helper: wrap an interactiveMessage payload and relay it.
 *
 * @param {object} sock - Baileys socket instance
 * @param {string} jid - Recipient JID
 * @param {object} interactiveContent - Plain JS interactiveMessage object
 * @param {object} [quoted] - Optional quoted message
 * @returns {Promise<object>} Generated message object
 */
async function _sendInteractive(sock, jid, interactiveContent, quoted) {
  const msg = generateWAMessageFromContent(jid, proto.Message.fromObject({
    messageContextInfo: {
      deviceListMetadata: {},
      deviceListMetadataVersion: 2,
      forwardedNewsletterMessageInfo: {
        newsletterJid: '0@newsletter',
        serverMessageId: -1,
        newsletterName: 'WazzapAgents',
      },
    },
    interactiveMessage: interactiveContent,
  }), {
    userJid: sock.user.id,
    ...(quoted ? { quoted } : {}),
  });
  logger.debug({ jid, messageId: msg.key.id }, 'relaying interactive message');
  await sock.relayMessage(jid, msg.message, { messageId: msg.key.id });
  return msg;
}

/**
 * Send quick-reply buttons.
 *
 * @param {object} sock - Baileys socket instance
 * @param {string} jid - Recipient JID
 * @param {string} body - Message body text
 * @param {Array<{id: string, displayText: string}>} buttons - Button definitions
 * @param {{footer?: string, title?: string, quoted?: object}} [options]
 * @returns {Promise<object>}
 * @example
 * await sendQuickReply(sock, jid, 'Pilih menu:', [
 *   { id: 'menu_1', displayText: 'Daftar Produk' },
 *   { id: 'menu_2', displayText: 'Hubungi CS' }
 * ], { title: 'Menu Utama', footer: 'Bot v1' });
 */
async function sendQuickReply(sock, jid, body, buttons, options = {}) {
  const nativeButtons = buttons.map((btn) => ({
    name: 'quick_reply',
    buttonParamsJson: JSON.stringify({ display_text: btn.displayText, id: btn.id }),
  }));
  return _sendInteractive(sock, jid, {
    body: { text: body },
    footer: { text: options.footer || '' },
    header: { title: options.title || '', hasMediaAttachment: false },
    nativeFlowMessage: { buttons: nativeButtons, messageParamsJson: '' },
  }, options.quoted);
}

/**
 * Send CTA URL buttons.
 *
 * @param {object} sock - Baileys socket instance
 * @param {string} jid - Recipient JID
 * @param {string} body - Message body text
 * @param {Array<{displayText: string, url: string, merchantUrl?: string}>} buttons
 * @param {{footer?: string, title?: string, quoted?: object}} [options]
 * @returns {Promise<object>}
 * @example
 * await sendUrlButtons(sock, jid, 'Kunjungi kami:', [
 *   { displayText: 'Website', url: 'https://example.com' }
 * ], { footer: 'Klik untuk buka' });
 */
async function sendUrlButtons(sock, jid, body, buttons, options = {}) {
  const nativeButtons = buttons.map((btn) => ({
    name: 'cta_url',
    buttonParamsJson: JSON.stringify({
      display_text: btn.displayText,
      url: btn.url,
      ...(btn.merchantUrl ? { merchant_url: btn.merchantUrl } : {}),
    }),
  }));
  return _sendInteractive(sock, jid, {
    body: { text: body },
    footer: { text: options.footer || '' },
    header: { title: options.title || '', hasMediaAttachment: false },
    nativeFlowMessage: { buttons: nativeButtons, messageParamsJson: '' },
  }, options.quoted);
}

/**
 * Send a single CTA copy-code button.
 *
 * @param {object} sock - Baileys socket instance
 * @param {string} jid - Recipient JID
 * @param {string} body - Message body text
 * @param {string} copyCode - Text copied to clipboard on tap
 * @param {string} [displayText='Copy Code'] - Button label
 * @param {{footer?: string, quoted?: object}} [options]
 * @returns {Promise<object>}
 * @example
 * await sendCopyCode(sock, jid, 'Kode promo Anda:', 'PROMO2024', 'Salin', {
 *   footer: 'Berlaku 7 hari'
 * });
 */
async function sendCopyCode(sock, jid, body, copyCode, displayText = 'Copy Code', options = {}) {
  return _sendInteractive(sock, jid, {
    body: { text: body },
    footer: { text: options.footer || '' },
    header: { hasMediaAttachment: false },
    nativeFlowMessage: {
      buttons: [{
        name: 'cta_copy',
        buttonParamsJson: JSON.stringify({ display_text: displayText, copy_code: copyCode }),
      }],
      messageParamsJson: '',
    },
  }, options.quoted);
}

/**
 * Send a mix of different button types (url, reply, copy, call) in one message.
 *
 * @param {object} sock - Baileys socket instance
 * @param {string} jid - Recipient JID
 * @param {string} body - Message body text
 * @param {Array<
 *   {type: 'url',   displayText: string, url: string} |
 *   {type: 'reply', displayText: string, id: string} |
 *   {type: 'copy',  displayText: string, copyCode: string} |
 *   {type: 'call',  displayText: string, phoneNumber: string}
 * >} buttons
 * @param {{footer?: string, title?: string, quoted?: object}} [options]
 * @returns {Promise<object>}
 * @example
 * await sendCombinedButtons(sock, jid, 'Pilih aksi:', [
 *   { type: 'reply', displayText: 'Konfirmasi', id: 'confirm' },
 *   { type: 'url',   displayText: 'Detail', url: 'https://example.com' },
 *   { type: 'call',  displayText: 'Telepon', phoneNumber: '+6281234567890' }
 * ]);
 */
async function sendCombinedButtons(sock, jid, body, buttons, options = {}) {
  const nativeButtons = buttons.map((btn) => {
    switch (btn.type) {
      case 'url':
        return { name: 'cta_url', buttonParamsJson: JSON.stringify({ display_text: btn.displayText, url: btn.url }) };
      case 'reply':
        return { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: btn.displayText, id: btn.id }) };
      case 'copy':
        return { name: 'cta_copy', buttonParamsJson: JSON.stringify({ display_text: btn.displayText, copy_code: btn.copyCode }) };
      case 'call':
        return { name: 'cta_call', buttonParamsJson: JSON.stringify({ display_text: btn.displayText, phone_number: btn.phoneNumber }) };
      default:
        return { name: btn.type, buttonParamsJson: JSON.stringify({ display_text: btn.displayText }) };
    }
  });
  return _sendInteractive(sock, jid, {
    body: { text: body },
    footer: { text: options.footer || '' },
    header: { title: options.title || '', hasMediaAttachment: false },
    nativeFlowMessage: { buttons: nativeButtons, messageParamsJson: '' },
  }, options.quoted);
}

/**
 * Send a single-select list (dropdown menu).
 * Uses listMessage which is supported directly via sock.sendMessage.
 *
 * @param {object} sock - Baileys socket instance
 * @param {string} jid - Recipient JID
 * @param {{
 *   title: string,
 *   buttonText: string,
 *   sections: Array<{title: string, rows: Array<{rowId: string, title: string, description?: string}>}>,
 *   footer?: string,
 *   description?: string
 * }} content
 * @param {{quoted?: object}} [options]
 * @returns {Promise<object>}
 * @example
 * await sendList(sock, jid, {
 *   title: 'Menu Restoran',
 *   buttonText: 'Lihat Menu',
 *   sections: [{
 *     title: 'Makanan',
 *     rows: [{ rowId: 'nasi_goreng', title: 'Nasi Goreng', description: 'Rp 25.000' }]
 *   }],
 *   footer: 'Order via chat'
 * });
 */
async function sendList(sock, jid, content, options = {}) {
  return sock.sendMessage(jid, {
    listMessage: {
      title: content.title,
      description: content.description || '',
      buttonText: content.buttonText,
      footerText: content.footer || '',
      listType: 1,
      sections: content.sections,
    },
  }, { quoted: options.quoted });
}

/**
 * Send a raw NativeFlow interactive message with pre-formatted buttons.
 *
 * @param {object} sock - Baileys socket instance
 * @param {string} jid - Recipient JID
 * @param {string} body - Message body text
 * @param {Array<{name: string, buttonParamsJson: string}>} buttons - Pre-formatted button array
 * @param {{footer?: string, header?: {title?: string, subtitle?: string}, quoted?: object}} [options]
 * @returns {Promise<object>}
 * @example
 * await sendNativeFlow(sock, jid, 'Choose:', [
 *   { name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Yes', id: 'yes' }) }
 * ], { footer: 'Tap to select' });
 */
async function sendNativeFlow(sock, jid, body, buttons, options = {}) {
  const header = {
    hasMediaAttachment: false,
    ...(options.header?.title ? { title: options.header.title } : {}),
    ...(options.header?.subtitle ? { subtitle: options.header.subtitle } : {}),
  };
  return _sendInteractive(sock, jid, {
    body: { text: body },
    footer: { text: options.footer || '' },
    header,
    nativeFlowMessage: { buttons, messageParamsJson: '' },
  }, options.quoted);
}

export {
  _sendInteractive,
  sendQuickReply,
  sendUrlButtons,
  sendCopyCode,
  sendCombinedButtons,
  sendList,
  sendNativeFlow,
};
