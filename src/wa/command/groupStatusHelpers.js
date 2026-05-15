/**
 * Pure helper for groupStatus command - no external imports.
 *
 * findRawMediaContent inspects a raw Baileys message object WITHOUT calling
 * normalizeMessageContent. This is needed because normalizeMessageContent
 * transforms imageMessage-with-caption into extendedTextMessage, destroying
 * the downloadable media object.
 *
 * Checks for imageMessage/videoMessage in:
 *   - Direct keys on rawMessage
 *   - Inside ephemeralMessage wrapper
 *   - Inside viewOnceMessage wrapper
 *   - Inside viewOnceMessageV2 wrapper
 *
 * @param {object|null|undefined} rawMessage - Raw Baileys message object (msg.message)
 * @returns {{ contentType: string, content: object }|null}
 */
function findRawMediaContent(rawMessage) {
  if (!rawMessage) return null;

  const mediaTypes = ['imageMessage', 'videoMessage'];

  // Direct keys
  for (const type of mediaTypes) {
    if (rawMessage[type]) {
      return { contentType: type, content: rawMessage[type] };
    }
  }

  // Ephemeral wrapper
  const ephemeralInner = rawMessage.ephemeralMessage?.message;
  if (ephemeralInner) {
    for (const type of mediaTypes) {
      if (ephemeralInner[type]) {
        return { contentType: type, content: ephemeralInner[type] };
      }
    }
  }

  // viewOnceMessage wrapper
  const viewOnceInner = rawMessage.viewOnceMessage?.message;
  if (viewOnceInner) {
    for (const type of mediaTypes) {
      if (viewOnceInner[type]) {
        return { contentType: type, content: viewOnceInner[type] };
      }
    }
  }

  // viewOnceMessageV2 wrapper
  const viewOnceV2Inner = rawMessage.viewOnceMessageV2?.message;
  if (viewOnceV2Inner) {
    for (const type of mediaTypes) {
      if (viewOnceV2Inner[type]) {
        return { contentType: type, content: viewOnceV2Inner[type] };
      }
    }
  }

  return null;
}

export { findRawMediaContent };
