"""Document thumbnail generation for WhatsApp previews.

WhatsApp displays a small preview/thumbnail for document attachments (PDF,
DOCX, etc.) only when a ``jpegThumbnail`` field is provided in the protocol
message.  PDF thumbnails are generated via *pypdfium2* (which bundles
PDFium/Chrome's PDF renderer, so no external poppler dependency is needed).
Image files sent as documents are thumbnailed with Pillow.

Office formats (DOCX, XLSX, PPTX, ODT, ODS, ODP) cannot be rendered on the
host because LibreOffice is only available inside the WazzapSubAgents Docker
container.  For these, a small placeholder icon is generated per MIME type.

All thumbnails are produced as small JPEG buffers (~50–100 KB) suitable for
inclusion in the Baileys ``DocumentMessage.jpegThumbnail`` field.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

try:
  from ..log import setup_logging
except ImportError:
  import sys
  from pathlib import Path as _Path
  sys.path.append(str(_Path(__file__).resolve().parent.parent.parent.parent))
  from bridge.log import setup_logging  # type: ignore

logger = setup_logging()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WhatsApp renders the document thumbnail at roughly 300×300 logical pixels.
# We produce a slightly larger image and let the client scale it down.
_THUMB_SIZE = 400

# Maximum thumbnail file size (bytes).  WhatsApp rejects very large thumbnails;
# 100 KB is generous enough for good quality while staying well under limits.
_MAX_THUMB_BYTES = 100 * 1024

# JPEG quality for thumbnail output.  75 is a good balance between quality
# and size for a 400×400 image.
_THUMB_JPEG_QUALITY = 75

# pypdfium2 render DPI for the first PDF page.  150 DPI gives a decent-quality
# thumbnail without being too slow or memory-hungry.
_PDF_RENDER_DPI = 150

# ---------------------------------------------------------------------------
# MIME-type → placeholder icon colour + label
# ---------------------------------------------------------------------------

_PLACEHOLDER_CONFIG: dict[str, tuple[str, str]] = {
  # mime_type_prefix  → (fill_colour_hex, short_label)
  "application/vnd.openxmlformats-officedocument.wordprocessingml": ("#2B579A", "DOCX"),
  "application/vnd.openxmlformats-officedocument.spreadsheetml":    ("#217346", "XLSX"),
  "application/vnd.openxmlformats-officedocument.presentationml":    ("#D24726", "PPTX"),
  "application/vnd.oasis.opendocument.text":                        ("#0E7FC2", "ODT"),
  "application/vnd.oasis.opendocument.spreadsheet":                  ("#0E7FC2", "ODS"),
  "application/vnd.oasis.opendocument.presentation":                 ("#0E7FC2", "ODP"),
  "application/msword":       ("#2B579A", "DOC"),
  "application/vnd.ms-excel":  ("#217346", "XLS"),
  "application/vnd.ms-powerpoint": ("#D24726", "PPT"),
  "application/rtf":          ("#7B5B3A", "RTF"),
  "text/plain":               ("#6B7280", "TXT"),
  "text/csv":                  ("#217346", "CSV"),
  "application/zip":           ("#6B7280", "ZIP"),
  "application/x-7z-compressed": ("#6B7280", "7Z"),
  "application/vnd.rar":       ("#6B7280", "RAR"),
  "application/gzip":          ("#6B7280", "GZ"),
  "application/x-tar":        ("#6B7280", "TAR"),
}

# ---------------------------------------------------------------------------
# Helper: generate a solid-colour placeholder icon with a short label
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
  """Convert ``#RRGGBB`` to ``(R, G, B)``."""
  h = hex_color.lstrip("#")
  return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _generate_placeholder(mime: str, ext: str) -> bytes:
  """Return a JPEG thumbnail buffer for file types we cannot render.

  The icon is a solid-colour square with the file extension (or a short
  MIME-type label) centred in white text on a coloured background.
  """
  # Find a matching placeholder config key
  config_entry = _PLACEHOLDER_CONFIG.get(mime)
  if config_entry is None:
    for key, val in _PLACEHOLDER_CONFIG.items():
      if mime.startswith(key):
        config_entry = val
        break

  if config_entry:
    fill_hex, label = config_entry
  else:
    fill_hex, label = "#6B7280", ext.upper()[:5] if ext else "FILE"

  bg_color = _hex_to_rgb(fill_hex)

  img = Image.new("RGB", (_THUMB_SIZE, _THUMB_SIZE), bg_color)
  draw = ImageDraw.Draw(img)

  # Try to use a TrueType font; fall back to the default bitmap font.
  font_size = max(24, _THUMB_SIZE // (len(label) + 1))
  font = _try_load_font(font_size)

  # Centre the label text
  bbox = draw.textbbox((0, 0), label, font=font)
  tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
  x = (_THUMB_SIZE - tw) // 2
  y = (_THUMB_SIZE - th) // 2
  draw.text((x, y), label, fill="white", font=font)

  buf = io.BytesIO()
  img.save(buf, format="JPEG", quality=_THUMB_JPEG_QUALITY)
  return buf.getvalue()


# ---------------------------------------------------------------------------
# Font loading (DejaVu Sans preferred; falls back to Pillow default)
# ---------------------------------------------------------------------------

_FONT_PATHS = [
  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
  "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _try_load_font(size: int) -> ImageFont.FreeFont | ImageFont.ImageFont:
  """Attempt to load a TrueType font at the given size; fall back to default."""
  for fp in _FONT_PATHS:
    try:
      return ImageFont.truetype(fp, size=size)
    except (OSError, IOError):
      continue
  return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Core: generate_document_thumbnail
# ---------------------------------------------------------------------------

def generate_document_thumbnail(
  file_path: str,
  mime: str,
) -> Optional[bytes]:
  """Generate a JPEG thumbnail buffer for *file_path*.

  Returns ``None`` if thumbnail generation fails for any reason (missing
  library, corrupt file, etc.).  The caller should simply send the document
  without a thumbnail in that case.

  Strategy by file type:
    - **PDF** – rendered from the first page via *pypdfium2* (bundled
      PDFium, no poppler required).
    - **Images** (jpeg, png, webp, gif, bmp, tiff, heic, avif) – resized
      via Pillow.
    - **Office / other** – a coloured placeholder icon, because LibreOffice
      is not available on the host.
  """
  if not file_path or not os.path.isfile(file_path):
    return None

  ext = os.path.splitext(file_path)[1].lower()

  try:
    # --- PDF ---------------------------------------------------------------
    if mime == "application/pdf" or ext == ".pdf":
      return _thumbnail_pdf(file_path)

    # --- Image files sent as documents --------------------------------------
    if mime.startswith("image/"):
      return _thumbnail_image(file_path)

    # --- Office & other document types: placeholder icon -------------------
    return _generate_placeholder(mime, ext)

  except Exception:
    logger.debug(
      "generate_document_thumbnail: failed for %s (mime=%s)",
      file_path, mime,
      exc_info=True,
    )
    return None


# ---------------------------------------------------------------------------
# PDF thumbnail via pypdfium2
# ---------------------------------------------------------------------------

def _thumbnail_pdf(file_path: str) -> Optional[bytes]:
  """Render the first page of a PDF and return a JPEG thumbnail."""
  try:
    import pypdfium2  # type: ignore
  except ImportError:
    logger.debug("pypdfium2 not available – skipping PDF thumbnail")
    return None

  try:
    pdf = pypdfium2.PdfDocument(file_path)
    if pdf.page_count < 1:
      return None
    page = pdf[0]
    # render() returns a list of PIL Images (one per page requested)
    result = page.render(scale=_PDF_RENDER_DPI / 72)
    pil_image = result.to_pil()

    thumb = _resize_to_thumbnail(pil_image)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=_THUMB_JPEG_QUALITY)
    data = buf.getvalue()
    pdf.close()
    return data if len(data) <= _MAX_THUMB_BYTES else None
  except Exception:
    logger.debug("pypdfium2 render failed for %s", file_path, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Image thumbnail via Pillow
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".avif"}


def _thumbnail_image(file_path: str) -> Optional[bytes]:
  """Resize an image file and return a JPEG thumbnail."""
  try:
    img = Image.open(file_path)
    # Convert to RGB (required for JPEG output)
    if img.mode not in ("RGB", "L"):
      img = img.convert("RGB")

    thumb = _resize_to_thumbnail(img)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=_THUMB_JPEG_QUALITY)
    data = buf.getvalue()
    return data if len(data) <= _MAX_THUMB_BYTES else None
  except Exception:
    logger.debug("Image thumbnail failed for %s", file_path, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Shared resize helper
# ---------------------------------------------------------------------------

def _resize_to_thumbnail(img: Image.Image) -> Image.Image:
  """Resize *img* so the longest side is ``_THUMB_SIZE`` (maintain AR)."""
  w, h = img.size
  if max(w, h) <= _THUMB_SIZE:
    return img
  ratio = _THUMB_SIZE / max(w, h)
  new_w = max(1, int(w * ratio))
  new_h = max(1, int(h * ratio))
  # LANCZOS is high-quality and fast enough for thumbnails
  return img.resize((new_w, new_h), Image.LANCZOS)