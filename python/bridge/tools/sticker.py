"""Sticker creation tool: generate image stickers with text overlay."""
from __future__ import annotations

import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = [
  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
  "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

_WATERMARK_TEXT = "Made with Vivy Ai."
_SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_SUPPORTED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".webm", ".3gp"}

# Output dir: data/media/ relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_OUTPUT_DIR = _PROJECT_ROOT / "data" / "media"


def create_sticker_file(
  media_path: str,
  upper_text: str | None = None,
  lower_text: str | None = None,
  font_size: int = 50,
) -> str:
  """
  Create a sticker from an image or video file with optional text overlay.

  - Images are opened directly.
  - Videos: first frame is extracted via ffmpeg.
  - Output is square-padded (transparent background) so any aspect ratio fits.
  - Text is uppercased, white with black outline, word-wrapped.
  - A small watermark "Made with Vivy Ai." is always added.

  Args:
    media_path: Absolute path to image or video file.
    upper_text: Text at the top (optional, will be uppercased).
    lower_text: Text at the bottom (optional, will be uppercased).
    font_size: Font size for text overlays (default 50).

  Returns:
    Absolute path to the saved PNG sticker.

  Raises:
    FileNotFoundError: If media file does not exist.
    ValueError: If file format is not supported.
  """
  if not os.path.exists(media_path):
    raise FileNotFoundError(f"Media not found: {media_path}")

  ext = Path(media_path).suffix.lower()
  if ext in _SUPPORTED_IMAGE_EXT:
    img = Image.open(media_path).convert("RGBA")
  elif ext in _SUPPORTED_VIDEO_EXT:
    img = _extract_first_frame(media_path)
    if img is None:
      raise ValueError(f"Failed to extract frame from video: {media_path}")
    img = img.convert("RGBA")
  else:
    supported = ", ".join(sorted(_SUPPORTED_IMAGE_EXT | _SUPPORTED_VIDEO_EXT))
    raise ValueError(f"Unsupported format: {ext}. Supported: {supported}")

  # Square-pad to max(w, h) with transparent background
  img = _square_pad(img)

  # Add text overlays
  if upper_text:
    upper_text = upper_text.upper()
  if lower_text:
    lower_text = lower_text.upper()

  font = _load_font(font_size)
  watermark_font = _load_font(max(font_size // 3, 14))

  img = _add_overlays(img, upper_text, lower_text, font, watermark_font)

  # Save to data/media/
  _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  short_id = uuid.uuid4().hex[:8]
  out_path = _OUTPUT_DIR / f"sticker_{short_id}.png"
  img.save(str(out_path), "PNG")
  return str(out_path)


def _square_pad(img: Image.Image) -> Image.Image:
  """Pad image to a square canvas using transparent background."""
  w, h = img.size
  if w == h:
    return img
  side = max(w, h)
  canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
  x = (side - w) // 2
  y = (side - h) // 2
  canvas.paste(img, (x, y))
  return canvas


def _extract_first_frame(video_path: str) -> Image.Image | None:
  """Extract the first frame from a video using ffmpeg."""
  try:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
      tmp_path = tmp.name
    cmd = [
      "ffmpeg", "-i", video_path,
      "-vframes", "1",
      "-ss", "0",
      "-y",
      tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
      return None
    frame = Image.open(tmp_path).copy()
    os.unlink(tmp_path)
    return frame
  except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
    return None


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  """Try known font paths; fall back to PIL default."""
  for path in _FONT_PATHS:
    if os.path.exists(path):
      try:
        return ImageFont.truetype(path, size)
      except Exception:
        continue
  return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
  """Word-wrap text to fit within max_width pixels."""
  outline_w = 3
  available = max_width - (outline_w * 2 + 20)
  dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
  words = text.split()
  lines: list[str] = []
  current = ""
  for word in words:
    test = (current + " " + word).strip()
    bbox = dummy_draw.textbbox((0, 0), test, font=font)
    if bbox[2] - bbox[0] <= available:
      current = test
    else:
      if current:
        lines.append(current)
      current = word
  if current:
    lines.append(current)
  return lines


def _draw_outlined_text(
  draw: ImageDraw.ImageDraw,
  position: tuple[int, int],
  text: str,
  font: ImageFont.ImageFont,
  outline_width: int = 3,
  anchor: str = "mm",
) -> None:
  """Draw text with a black outline followed by white fill."""
  x, y = position
  for dx in range(-outline_width, outline_width + 1):
    for dy in range(-outline_width, outline_width + 1):
      if dx != 0 or dy != 0:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255), anchor=anchor)
  draw.text(position, text, font=font, fill=(255, 255, 255, 255), anchor=anchor)


def _add_overlays(
  img: Image.Image,
  upper_text: str | None,
  lower_text: str | None,
  font: ImageFont.ImageFont,
  watermark_font: ImageFont.ImageFont,
) -> Image.Image:
  """Add upper/lower text blocks and watermark to image."""
  draw = ImageDraw.Draw(img)
  width, height = img.size
  outline_w = 3
  padding = 20
  line_spacing = int(_get_font_size(font) * 1.3)

  # Upper text — grows downward from top
  if upper_text:
    lines = _wrap_text(upper_text, font, width)
    bbox = draw.textbbox((0, 0), lines[0] if lines else "", font=font)
    text_h = bbox[3] - bbox[1]
    y = padding + text_h // 2
    for line in lines:
      _draw_outlined_text(draw, (width // 2, y), line, font, outline_w, "mm")
      y += line_spacing

  # Lower text — grows upward from bottom
  if lower_text:
    lines = _wrap_text(lower_text, font, width)
    y = height - padding
    for line in reversed(lines):
      _draw_outlined_text(draw, (width // 2, y), line, font, outline_w, "mb")
      y -= line_spacing

  # Watermark — small text at bottom-right
  wm_lines = _wrap_text(_WATERMARK_TEXT, watermark_font, width)
  wm_y = height - 6
  for line in reversed(wm_lines):
    _draw_outlined_text(draw, (width - 8, wm_y), line, watermark_font, 1, "rb")
    wm_bbox = draw.textbbox((0, 0), line, font=watermark_font)
    wm_y -= (wm_bbox[3] - wm_bbox[1]) + 2

  return img


def _get_font_size(font: ImageFont.ImageFont) -> int:
  """Best-effort font size extraction."""
  if hasattr(font, "size"):
    return int(font.size)
  return 12
