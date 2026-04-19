import sharp from 'sharp';
import path from 'path';
import fs from 'fs-extra';
import { randomUUID } from 'crypto';
import config from '../config.js';

const STICKER_SIZE = 512;
const OUTPUT_DIR = config.stickersDir;

const SUPPORTED_IMAGE_EXT = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);

async function createStickerFile(mediaPath, upperText = null, lowerText = null) {
  const ext = path.extname(mediaPath).toLowerCase();

  if (!SUPPORTED_IMAGE_EXT.has(ext)) {
    throw new Error(`Unsupported format: ${ext}`);
  }

  // Keep aspect ratio, center on transparent 512x512 canvas,
  // and allow both downscale/upscale so users don't need exact 512x512 input.
  const img = sharp(mediaPath).resize(STICKER_SIZE, STICKER_SIZE, {
    fit: 'contain',
    withoutEnlargement: false,
    background: { r: 0, g: 0, b: 0, alpha: 0 },
  });

  await fs.ensureDir(OUTPUT_DIR);
  const shortId = randomUUID().slice(0, 8);
  const outPath = path.join(OUTPUT_DIR, `sticker_${shortId}.webp`);

  await img.webp({ quality: 95 }).toFile(outPath);
  return outPath;
}

export { createStickerFile };
