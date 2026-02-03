import { finished } from 'stream';
import { promisify } from 'util';
import logger from './logger.js';

const streamFinished = promisify(finished);

export async function streamToBuffer(stream) {
  const chunks = [];

  // Handle AsyncIterable (Baileys media stream)
  if (stream && typeof stream[Symbol.asyncIterator] === 'function') {
    for await (const chunk of stream) {
      chunks.push(chunk);
    }
    return Buffer.concat(chunks);
  }

  // Fallback: Node.js stream
  return new Promise((resolve, reject) => {
    if (!stream || typeof stream.on !== 'function') {
      return reject(new Error('Invalid stream'));
    }
    stream.on('data', (chunk) => chunks.push(chunk));
    stream.on('error', (err) => {
      logger.error({ err }, 'stream error');
      reject(err);
    });
    stream.on('end', () => resolve(Buffer.concat(chunks)));
    streamFinished(stream).catch(reject);
  });
}

export default {
  streamToBuffer,
};
