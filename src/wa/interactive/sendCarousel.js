/**
 * sendCarousel.js — Carousel / swipeable Cards messages.
 *
 * Carousel is an interactiveMessage with carouselMessage inside.
 * Must be sent via generateWAMessageFromContent + sock.relayMessage —
 * sock.sendMessage does not support this content shape.
 */
import { _sendInteractive } from './sendInteractive.js';

/**
 * Send a carousel message with swipeable cards.
 *
 * @param {object} sock - Baileys socket instance
 * @param {string} jid - Recipient JID
 * @param {Array<{
 *   image?: {url: string} | Buffer,
 *   video?: {url: string} | Buffer,
 *   title?: string,
 *   body?: string,
 *   footer?: string,
 *   buttons?: Array<{name: string, buttonParamsJson: string}>
 * }>} cards - Carousel card definitions
 * @param {{text?: string, title?: string, subtitle?: string, footer?: string, quoted?: object}} [options]
 * @returns {Promise<object>}
 * @example
 * await sendCarousel(sock, jid, [
 *   {
 *     image: { url: 'https://example.com/p1.jpg' },
 *     title: 'Produk A',
 *     body: 'Deskripsi A',
 *     footer: 'Rp 100.000',
 *     buttons: [{ name: 'quick_reply', buttonParamsJson: JSON.stringify({ display_text: 'Beli', id: 'buy_a' }) }]
 *   }
 * ], { text: 'Produk Unggulan', footer: 'Swipe untuk lihat lebih' });
 */
async function sendCarousel(sock, jid, cards, options = {}) {
  const mappedCards = cards.map((card) => {
    const header = {};
    if (card.image) header.imageMessage = { url: card.image.url ?? card.image };
    if (card.video) header.videoMessage = { url: card.video.url ?? card.video };
    if (card.title) header.title = card.title;

    return {
      header,
      body: { text: typeof card.body === 'string' ? card.body : (card.body?.text || '') },
      footer: { text: typeof card.footer === 'string' ? card.footer : (card.footer?.text || '') },
      nativeFlowMessage: {
        buttons: (card.buttons || []).map((btn) => ({
          name: btn.name,
          buttonParamsJson: btn.buttonParamsJson,
        })),
      },
    };
  });

  return _sendInteractive(sock, jid, {
    body: { text: options.text || '' },
    ...(options.footer ? { footer: { text: options.footer } } : {}),
    carouselMessage: { cards: mappedCards },
  }, options.quoted);
}

export { sendCarousel };
