import path from 'path';
import fs from 'fs-extra';
import { downloadContentFromMessage } from 'baileys';
import logger from './logger.js';
import config from './config.js';
import { streamToFile } from './utils.js';

function isPathWithin(basePath, candidatePath) {
  const relative = path.relative(basePath, candidatePath);
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
}

async function resolveAllowedAttachmentPath(rawPath, actionError) {
  if (typeof rawPath !== 'string' || !rawPath.trim()) {
    throw actionError('invalid_target', 'attachment path is required');
  }
  const candidate = path.resolve(rawPath.trim());
  if (!await fs.pathExists(candidate)) {
    throw actionError('not_found', `attachment not found: ${rawPath}`);
  }
  const [mediaDirRealPath, candidateRealPath] = await Promise.all([
    fs.realpath(config.mediaDir),
    fs.realpath(candidate),
  ]);
  if (!isPathWithin(mediaDirRealPath, candidateRealPath)) {
    throw actionError('invalid_target', `attachment path must be inside media dir: ${config.mediaDir}`);
  }
  const stat = await fs.stat(candidateRealPath);
  if (!stat.isFile()) {
    throw actionError('invalid_target', 'attachment path must point to a file');
  }
  return candidateRealPath;
}

function inferExtension(mime) {
  const normalized = normalizeMime(mime);
  if (!normalized) return 'bin';
  if (normalized.includes('jpeg')) return 'jpg';
  if (normalized.includes('png')) return 'png';
  if (normalized.includes('gif')) return 'gif';
  if (normalized.includes('webp')) return 'webp';
  if (normalized.includes('mp4')) return 'mp4';
  if (normalized.includes('mp3')) return 'mp3';
  if (normalized.includes('ogg')) return 'ogg';
  if (normalized.includes('pdf')) return 'pdf';
  if (normalized.includes('zip')) return 'zip';
  return normalized.split('/').pop() || 'bin';
}

function normalizeMime(mime) {
  if (typeof mime !== 'string') return null;
  const normalized = mime.split(';')[0].trim().toLowerCase();
  return normalized || null;
}

function detectMimeFromHeader(header) {
  if (!Buffer.isBuffer(header) || header.length === 0) return null;

  if (
    header.length >= 12
    && header.toString('ascii', 0, 4) === 'RIFF'
    && header.toString('ascii', 8, 12) === 'WEBP'
  ) return 'image/webp';
  if (
    header.length >= 8
    && header[0] === 0x89
    && header[1] === 0x50
    && header[2] === 0x4E
    && header[3] === 0x47
    && header[4] === 0x0D
    && header[5] === 0x0A
    && header[6] === 0x1A
    && header[7] === 0x0A
  ) return 'image/png';
  if (header.length >= 3 && header[0] === 0xFF && header[1] === 0xD8 && header[2] === 0xFF) return 'image/jpeg';
  const gifMagic = header.toString('ascii', 0, 6);
  if (gifMagic === 'GIF87a' || gifMagic === 'GIF89a') return 'image/gif';
  if (header.length >= 4 && header.toString('ascii', 0, 4) === '%PDF') return 'application/pdf';
  if (
    header.length >= 4
    && header[0] === 0x50
    && header[1] === 0x4B
    && (header[2] === 0x03 || header[2] === 0x05 || header[2] === 0x07)
    && (header[3] === 0x04 || header[3] === 0x06 || header[3] === 0x08)
  ) return 'application/zip';
  if (header.length >= 4 && header.toString('ascii', 0, 4) === 'OggS') return 'audio/ogg';
  if (header.length >= 3 && header.toString('ascii', 0, 3) === 'ID3') return 'audio/mp3';
  if (header.length >= 2 && header[0] === 0xFF && (header[1] & 0xE0) === 0xE0) return 'audio/mp3';
  if (header.length >= 8 && header.toString('ascii', 4, 8) === 'ftyp') return 'video/mp4';

  return null;
}

async function readFileHeader(filepath, bytes = 16) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    const stream = fs.createReadStream(filepath, { start: 0, end: bytes - 1 });
    stream.on('data', (chunk) => chunks.push(chunk));
    stream.on('error', reject);
    stream.on('end', () => resolve(Buffer.concat(chunks)));
  });
}

async function detectMimeFromFile(filepath) {
  try {
    const header = await readFileHeader(filepath, 16);
    return detectMimeFromHeader(header);
  } catch (err) {
    logger.debug({ err, filepath }, 'failed to inspect saved media header');
    return null;
  }
}

function shouldRetryStickerAsImage(err) {
  const message = String(err?.message || '').toLowerCase();
  if (!message) return false;
  return (
    message.includes('bad decrypt')
    || message.includes('unable to authenticate data')
    || message.includes('wrong final block length')
    || message.includes('mac check failed')
    || message.includes('failed to decrypt')
  );
}

async function downloadMediaToFile(content, mediaKind, filepath, withTimeout) {
  const stream = await withTimeout(
    downloadContentFromMessage(content, mediaKind),
    config.downloadTimeoutMs,
    `downloadContentFromMessage(${mediaKind})`
  );
  return withTimeout(
    streamToFile(stream, filepath),
    config.downloadTimeoutMs,
    `streamToFile(${mediaKind})`
  );
}

function mapMediaKind(contentType) {
  if (contentType === 'imageMessage') return 'image';
  if (contentType === 'videoMessage') return 'video';
  if (contentType === 'audioMessage') return 'audio';
  if (contentType === 'documentMessage') return 'document';
  if (contentType === 'stickerMessage') return 'sticker';
  return 'unknown';
}

async function saveMedia(contentType, content, messageId, withTimeout) {
  const kind = mapMediaKind(contentType);
  if (kind === 'unknown') return null;
  const declaredMime = normalizeMime(content?.mimetype);
  let mime = declaredMime || (kind === 'sticker' ? 'image/webp' : 'application/octet-stream');
  let ext = inferExtension(mime);
  let filename = `${messageId}_${kind}.${ext}`;
  let filepath = path.join(config.mediaDir, filename);
  let usedImageFallback = false;

  let size;
  try {
    size = await downloadMediaToFile(content, kind, filepath, withTimeout);
  } catch (err) {
    if (kind !== 'sticker' || !shouldRetryStickerAsImage(err)) throw err;
    logger.warn({ err, messageId }, 'sticker decrypt failed with kind=sticker, retry as image');
    await fs.remove(filepath).catch(() => {});
    usedImageFallback = true;
    size = await downloadMediaToFile(content, 'image', filepath, withTimeout);
  }

  const shouldUseDetectedMime = !declaredMime || declaredMime === 'application/octet-stream' || usedImageFallback;
  const detectedMime = shouldUseDetectedMime ? await detectMimeFromFile(filepath) : null;
  if (detectedMime) {
    mime = detectedMime;
    ext = inferExtension(mime);
    const detectedFilename = `${messageId}_${kind}.${ext}`;
    const detectedFilepath = path.join(config.mediaDir, detectedFilename);
    if (detectedFilepath !== filepath) {
      await fs.move(filepath, detectedFilepath, { overwrite: true });
      filename = detectedFilename;
      filepath = detectedFilepath;
    }
  }

  return {
    kind,
    mime,
    fileName: filename,
    size,
    path: filepath,
    isAnimated: Boolean(content?.isAnimated),
  };
}

export {
  isPathWithin,
  resolveAllowedAttachmentPath,
  inferExtension,
  normalizeMime,
  detectMimeFromHeader,
  readFileHeader,
  detectMimeFromFile,
  shouldRetryStickerAsImage,
  downloadMediaToFile,
  mapMediaKind,
  saveMedia,
};
