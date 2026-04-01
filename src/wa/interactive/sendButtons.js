import { getSock } from '../connection.js';
import { actionError } from '../actions.js';
import { sendInteractive } from './sendInteractive.js';

// ---------------------------------------------------------------------------
// Button param types
// ---------------------------------------------------------------------------

/**
 * @typedef {object} QuickReplyParams
 * @property {string} display_text - Label text shown on the button
 * @property {string} id - Unique identifier returned on tap
 */

/**
 * @typedef {object} CtaUrlParams
 * @property {string} display_text - Label text shown on the button
 * @property {string} url - URL to open when tapped
 * @property {string} merchant_url - Merchant/business URL for verification
 */

/**
 * @typedef {object} CtaCopyParams
 * @property {string} display_text - Label text shown on the button
 * @property {string} id - Unique identifier for the button
 * @property {string} copy_code - Text copied to clipboard on tap
 */

/**
 * @typedef {object} CtaCallParams
 * @property {string} display_text - Label text shown on the button
 * @property {string} id - Unique identifier for the button
 * @property {string} phone_number - Phone number to call on tap
 */

/**
 * @typedef {object} SingleSelectSection
 * @property {string} title - Section title
 * @property {Array<{title: string, description?: string, id: string}>} rows - Row items
 */

/**
 * @typedef {object} SingleSelectParams
 * @property {string} title - Menu dropdown title
 * @property {SingleSelectSection[]} sections - Menu sections with row items
 */

/**
 * @typedef {object} InteractiveButton
 * @property {'quick_reply'|'cta_url'|'cta_copy'|'cta_call'|'single_select'} name - Button type
 * @property {QuickReplyParams|CtaUrlParams|CtaCopyParams|CtaCallParams|SingleSelectParams} buttonParams
 *   Parameters for the button. Will be JSON.stringify'd internally — pass a
 *   plain object, NOT a pre-stringified value.
 */

// ---------------------------------------------------------------------------
// sendButtons
// ---------------------------------------------------------------------------

/**
 * Send an interactive button message (max 3 buttons) or a menu dropdown.
 *
 * @param {object} payload
 * @param {string} payload.chatId - Target JID
 * @param {string} payload.text - Body text
 * @param {string} payload.footer - Footer text
 * @param {InteractiveButton[]} payload.buttons - Buttons array (max 3)
 * @returns {Promise<{messageId: string}>}
 */
async function sendButtons({ chatId, text, footer, buttons }) {
  const sock = getSock();
  if (!sock) throw actionError('send_failed', 'WhatsApp socket not ready');
  if (!chatId) throw actionError('invalid_target', 'chatId is required');
  if (!Array.isArray(buttons) || buttons.length === 0) {
    throw actionError('invalid_target', 'buttons must be a non-empty array');
  }

  const nativeButtons = buttons.map((btn) => ({
    name: btn.name,
    buttonParamsJson: JSON.stringify(btn.buttonParams)
  }));

  const interactiveContent = {
    body: { text: text || '' },
    footer: { text: footer || '' },
    nativeFlowMessage: { buttons: nativeButtons, messageVersion: 1 }
  };

  const msg = await sendInteractive(sock, chatId, interactiveContent);

  return { messageId: msg.key.id };
}

export { sendButtons };
