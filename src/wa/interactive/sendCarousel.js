/**
 * sendCarousel.js — Carousel / swipeable Cards messages.
 *
 * Cards are passed directly to `sock.sendMessage` without proto imports.
 */

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
  const content = {
    ...(options.text ? { text: options.text } : {}),
    ...(options.title ? { title: options.title } : {}),
    ...(options.subtitle ? { subtitle: options.subtitle } : {}),
    ...(options.footer ? { footer: options.footer } : {}),
    cards,
  };
  return sock.sendMessage(jid, content, { quoted: options.quoted });
}

export { sendCarousel };
