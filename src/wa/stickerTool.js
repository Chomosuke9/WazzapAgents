import sharp from 'sharp';
import path from 'path';
import fs from 'fs-extra';
import { randomUUID } from 'crypto';
import config from '../config.js';
import logger from '../logger.js';

const STICKER_SIZE = 512;
const OUTPUT_DIR = config.stickersDir;

const SUPPORTED_IMAGE_EXT = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);
const SUPPORTED_VIDEO_EXT = new Set(['.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm', '.3gp']);

async function squarePad(img) {
  const { width, height } = await img.metadata();
  if (width === height) return img;
  const side = Math.max(width, height);
  const canvas = await sharp({
    create: {
      width: side,
      height: side,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 },
    },
  }).png().toBuffer();
  const inputBuffer = await img.png().toBuffer();
  const x = Math.floor((side - width) / 2);
  const y = Math.floor((side - height) / 2);
  return sharp(canvas)
    .composite([{ input: inputBuffer, left: x, top: y }])
    .png();
}

async function createStickerFile(mediaPath, upperText = null, lowerText = null) {
  const ext = path.extname(mediaPath).toLowerCase();

  if (!SUPPORTED_IMAGE_EXT.has(ext) && !SUPPORTED_VIDEO_EXT.has(ext)) {
    throw new Error(`Unsupported format: ${ext}`);
  }

  let img = sharp(mediaPath);
  img = img.resize(1024, 1024, { fit: 'inside', withoutEnlargement: true });
  img = await squarePad(img);
  img = img.resize(STICKER_SIZE, STICKER_SIZE, { fit: 'fill' });

  await fs.ensureDir(OUTPUT_DIR);
  const shortId = randomUUID().slice(0, 8);
  const outPath = path.join(OUTPUT_DIR, `sticker_${shortId}.webp`);

  await img.webp({ quality: 95 }).toFile(outPath);
  return outPath;
}

export { createStickerFile };