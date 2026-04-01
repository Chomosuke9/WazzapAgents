import { getSock } from '../connection.js';
import { actionError } from '../actions.js';
import { sendInteractive } from './sendInteractive.js';

// ---------------------------------------------------------------------------
// Card types
// ---------------------------------------------------------------------------

/**
 * @typedef {object} CardHeader
 * @property {object} [imageMessage] - Optional image for the card header
 * @property {string} [imageMessage.url] - Image URL or file path
 * @property {string} [imageMessage.mimetype] - Image MIME type (e.g. 'image/jpeg')
 */

/**
 * @typedef {object} CardBody
 * @property {string} text - Card body text
 */

/**
 * @typedef {object} CardFooter
 * @property {string} text - Card footer text
 */

/**
 * @typedef {import('./sendButtons.js').InteractiveButton} InteractiveButton
 */

/**
 * @typedef {object} CarouselCard
 * @property {CardHeader} [header] - Optional header (supports imageMessage)
 * @property {CardBody} body - Card body
 * @property {CardFooter} [footer] - Optional card footer
 * @property {InteractiveButton[]} buttons - Buttons specific to this card
 */

// ---------------------------------------------------------------------------
// sendCarousel
// ---------------------------------------------------------------------------

/**
 * Send a carousel message with swipeable cards.
 *
 * @param {object} payload
 * @param {string} payload.chatId - Target JID
 * @param {string} payload.text - Body text displayed above the carousel
 * @param {CarouselCard[]} payload.cards - Array of carousel cards
 * @returns {Promise<{messageId: string}>}
 */
async function sendCarousel({ chatId, text, cards }) {
  const sock = getSock();
  if (!sock) throw actionError('send_failed', 'WhatsApp socket not ready');
  if (!chatId) throw actionError('invalid_target', 'chatId is required');
  if (!Array.isArray(cards) || cards.length === 0) {
    throw actionError('invalid_target', 'cards must be a non-empty array');
  }

  const mappedCards = cards.map((card) => ({
    header: card.header || {},
    body: card.body || { text: '' },
    footer: card.footer || { text: '' },
    nativeFlowMessage: {
      buttons: (card.buttons || []).map((btn) => ({
        name: btn.name,
        buttonParamsJson: JSON.stringify(btn.buttonParams)
      }))
    }
  }));

  const interactiveContent = {
    body: { text: text || '' },
    carouselMessage: { cards: mappedCards }
  };

  const msg = await sendInteractive(sock, chatId, interactiveContent);

  return { messageId: msg.key.id };
}

export { sendCarousel };
