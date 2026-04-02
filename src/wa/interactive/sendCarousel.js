/**
 * sendCarousel.js — Carousel / swipeable Cards messages.
 *
 * Carousel is an interactiveMessage with carouselMessage inside.
 * Requires binary XML node injection via additionalNodes on relayMessage.
 * Unlike NativeFlow buttons (type=native_flow), carousel needs type=carousel
 * in the interactive node.
 */
import { proto, generateWAMessageFromContent, isJidGroup } from 'baileys';
import logger from '../../logger.js';

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
function buildCarouselNodes(jid) {
  const nodes = [
    {
      tag: 'biz',
      attrs: {},
      content: [
        {
          tag: 'interactive',
          attrs: { type: 'carousel', v: '1' },
          content: [],
        },
      ],
    },
  ];
  if (!isJidGroup(jid)) {
    nodes.push({ tag: 'bot', attrs: { biz_bot: '1' } });
  }
  return nodes;
}

async function sendCarousel(sock, jid, cards, options = {}) {
  const mappedCards = cards.map((card) => {
    const headerFields = { hasMediaAttachment: false };
    if (card.image) {
      headerFields.hasMediaAttachment = true;
      headerFields.imageMessage = { url: card.image.url ?? card.image };
    } else if (card.video) {
      headerFields.hasMediaAttachment = true;
      headerFields.videoMessage = { url: card.video.url ?? card.video };
    }
    if (card.title) headerFields.title = card.title;

    return proto.Message.InteractiveMessage.create({
      header: proto.Message.InteractiveMessage.Header.create(headerFields),
      body: proto.Message.InteractiveMessage.Body.create({
        text: typeof card.body === 'string' ? card.body : (card.body?.text || ''),
      }),
      footer: proto.Message.InteractiveMessage.Footer.create({
        text: typeof card.footer === 'string' ? card.footer : (card.footer?.text || ''),
      }),
      nativeFlowMessage: proto.Message.InteractiveMessage.NativeFlowMessage.create({
        buttons: card.buttons || [],
      }),
    });
  });

  const interactiveContent = proto.Message.InteractiveMessage.create({
    body: proto.Message.InteractiveMessage.Body.create({ text: options.text || '' }),
    ...(options.footer ? {
      footer: proto.Message.InteractiveMessage.Footer.create({ text: options.footer }),
    } : {}),
    carouselMessage: proto.Message.InteractiveMessage.CarouselMessage.create({ cards: mappedCards }),
  });

  const msg = generateWAMessageFromContent(jid, {
    viewOnceMessage: {
      message: {
        interactiveMessage: interactiveContent,
      },
    },
  }, {
    userJid: sock.user.id,
    ...(options.quoted ? { quoted: options.quoted } : {}),
  });

  logger.debug({ jid, messageId: msg.key.id }, 'relaying carousel message');
  await sock.relayMessage(jid, msg.message, {
    messageId: msg.key.id,
    additionalNodes: buildCarouselNodes(jid),
  });

  return msg;
}

export { sendCarousel };
