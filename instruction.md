Create a /sticker <upper_text>#<lower_text> command.
basically they need to send image with this command as a caption OR reply a message with an image with this command.
video should also work(getting first frame of video)

also create a tool for llm2 to create a sticker. the tools should look like this:

#i write it as normal function, you need to convert it into llm structure.
def create_sticker(id: message id, upper_text=None, lower_text=None, font_size=50):
"""Create a sticker and send it to...
this is NOT for your daily usage, use it when explicitly asked to create it.
"""
#basically like that, correct that later.

# bla bla bla, make it by yourself.


here is the function you need to create a sticker. edit it to send directly to whatsapp.

#!/usr/bin/env python3
"""
create_sticker(media, upper_text, lower_text, font_size)
Generate image sticker dengan text overlay (uppercase, white dengan black outline)
Support foto & video (extract first frame)
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess
import tempfile

def create_sticker(media, upper_text=None, lower_text=None, font_size=50):
    """
    Create sticker dari media dengan text overlay
    
    Args:
        media: Path ke file image/video
        upper_text: Text di atas (akan di-uppercase), default None
        lower_text: Text di bawah (akan di-uppercase), default None
        font_size: Ukuran font (default 50)
    
    Returns:
        PIL.Image object atau None jika error
    
    Raises:
        FileNotFoundError: Jika media tidak ditemukan
        ValueError: Jika format tidak support
    """
    
    # Validate input
    if not os.path.exists(media):
        raise FileNotFoundError(f"Media tidak ditemukan: {media}")
    
    file_ext = Path(media).suffix.lower()
    
    # Step 1: Determine media type & extract image
    print(f"[*] Processing: {media}")
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
        # Itu foto
        print(f"[*] Detected image format: {file_ext}")
        img = Image.open(media).convert('RGB')
        
    elif file_ext in ['.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm', '.3gp']:
        # Itu video - extract frame pertama
        print(f"[*] Detected video format: {file_ext}")
        img = _extract_first_frame(media)
        if img is None:
            raise ValueError(f"Gagal extract frame dari video: {media}")
    else:
        raise ValueError(f"Format tidak support: {file_ext}. Gunakan: jpg, png, mp4, mov, avi, mkv, webm, 3gp")
    
    # Step 2: Add text overlays
    print(f"[*] Adding text overlays...")
    upper_text = upper_text.upper() if upper_text else None
    lower_text = lower_text.upper() if lower_text else None
    
    img = _add_text_with_outline(img, upper_text, lower_text, font_size)
    
    print(f"[✓] Sticker created successfully!")
    return img

def _extract_first_frame(video_path):
    """
    Extract frame pertama dari video menggunakan ffmpeg
    
    Args:
        video_path: Path ke video file
    
    Returns:
        PIL.Image object atau None
    """
    try:
        # Create temp file untuk frame
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Run ffmpeg: ambil 1 frame di detik ke-0
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vframes', '1',  # Extract 1 frame only
            '-ss', '0',       # Di detik ke-0 (frame pertama)
            '-y',             # Overwrite tanpa ask
            tmp_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"[!] FFmpeg error: {result.stderr.decode()}")
            return None
        
        # Load image
        img = Image.open(tmp_path).convert('RGB')
        
        # Cleanup temp file
        os.unlink(tmp_path)
        
        return img
    
    except FileNotFoundError:
        print("[!] ffmpeg not found. Install dengan: apt-get install ffmpeg")
        return None
    except subprocess.TimeoutExpired:
        print("[!] FFmpeg timeout (video terlalu besar?)")
        return None
    except Exception as e:
        print(f"[!] Error extracting frame: {e}")
        return None

def _wrap_text(text, font, max_width):
    """
    Wrap text ke multiple lines agar fit di width tertentu
    
    Args:
        text: Text yang ingin di-wrap
        font: PIL Font object
        max_width: Maximum width available
    
    Returns:
        List of text lines
    """
    
    outline_width = 3
    available_width = max_width - (outline_width * 2 + 20)  # 20px padding
    
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        test_line = (current_line + " " + word).strip()
        bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox(
            (0, 0), test_line, font=font
        )
        text_width = bbox[2] - bbox[0]
        
        if text_width <= available_width:
            current_line = test_line
        else:
            # Word terlalu panjang atau line penuh, buat line baru
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines

def _add_text_with_outline(img, upper_text, lower_text, font_size):
    """
    Add text overlay dengan outline (stroke)
    Anchor: upper text from top edge, lower text from bottom edge
    Text wrapping: long text dibagi ke multiple lines
    
    Args:
        img: PIL Image object
        upper_text: Text untuk top
        lower_text: Text untuk bottom
        font_size: Font size
    
    Returns:
        PIL Image object dengan text overlay
    """
    
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    
    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        print("[!] Font tidak ditemukan, menggunakan default")
        font = ImageFont.load_default()
    
    # Outline thickness
    outline_width = 3
    
    # Helper function: draw text dengan outline
    def draw_outlined_text(draw, text, position, font, fill_color, outline_color, outline_width, anchor):
        """Draw text dengan black outline"""
        x, y = position
        
        # Draw outline (di 8 arah sekitar text)
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                if adj_x != 0 or adj_y != 0:
                    draw.text(
                        (x + adj_x, y + adj_y),
                        text,
                        font=font,
                        fill=outline_color,
                        anchor=anchor
                    )
        
        # Draw text utama (putih)
        draw.text(
            position,
            text,
            font=font,
            fill=fill_color,
            anchor=anchor
        )
    
    # Text properties
    fill_color = (255, 255, 255)      # Putih
    outline_color = (0, 0, 0)         # Hitam
    padding = 20
    line_spacing = int(font_size * 1.3)  # Line spacing = 130% dari font size
    
    # Draw upper text (anchor from TOP edge, grow downward)
    if upper_text:
        upper_lines = _wrap_text(upper_text, font, width)
        
        if upper_lines:
            print(f"[*] Upper text wrapped to {len(upper_lines)} line(s)")
            # Calculate first line Y position
            bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox(
                (0, 0), upper_lines[0], font=font
            )
            text_height = bbox[3] - bbox[1]
            current_y = padding + (text_height // 2)  # Center first line vertically at padding
            
            for line in upper_lines:
                upper_pos = (width // 2, current_y)
                draw_outlined_text(
                    draw,
                    line,
                    upper_pos,
                    font,
                    fill_color,
                    outline_color,
                    outline_width,
                    anchor="mm"  # middle-middle: proper center anchor
                )
                current_y += line_spacing
    
    # Draw lower text (anchor from BOTTOM edge, grow upward)
    if lower_text:
        lower_lines = _wrap_text(lower_text, font, width)
        
        if lower_lines:
            print(f"[*] Lower text wrapped to {len(lower_lines)} line(s)")
            # For bottom anchor, we need to draw from bottom upward
            current_y = height - padding
            
            # Draw lines in reverse order (bottom to top)
            for line in reversed(lower_lines):
                lower_pos = (width // 2, current_y)
                draw_outlined_text(
                    draw,
                    line,
                    lower_pos,
                    font,
                    fill_color,
                    outline_color,
                    outline_width,
                    anchor="mb"  # middle-bottom
                )
                current_y -= line_spacing
    
    return img

# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEXT STICKER GENERATOR - WazzapAgents")
    print("=" * 70)
    
    # Create demo image untuk test
    demo_image = "/home/claude/demo_input.jpg"
    if not os.path.exists(demo_image):
        print("[*] Creating demo image...")
        demo_img = Image.new('RGB', (800, 600), color=(73, 109, 137))
        demo_img.save(demo_image)
        print(f"[✓] Created: {demo_image}")
    
    # Test fungsi
    try:
        # Test 1: Both texts
        print("\n[TEST 1] With both upper and lower text")
        result = create_sticker(
            media=demo_image,
            upper_text="This is SPARTA",
            lower_text="Bottom text here",
            font_size=50
        )
        result.save("/mnt/user-data/outputs/sticker_demo_both.jpg")
        print(f"[✓] Saved: sticker_demo_both.jpg")
        
        # Test 2: Only upper text
        print("\n[TEST 2] With only upper text")
        result = create_sticker(
            media=demo_image,
            upper_text="Only Top Text",
            font_size=50
        )
        result.save("/mnt/user-data/outputs/sticker_demo_upper.jpg")
        print(f"[✓] Saved: sticker_demo_upper.jpg")
        
        # Test 3: Only lower text
        print("\n[TEST 3] With only lower text")
        result = create_sticker(
            media=demo_image,
            lower_text="Only Bottom Text",
            font_size=50
        )
        result.save("/mnt/user-data/outputs/sticker_demo_lower.jpg")
        print(f"[✓] Saved: sticker_demo_lower.jpg")
        
        # Test 4: No text (plain image)
        print("\n[TEST 4] No text (plain image)")
        result = create_sticker(media=demo_image)
        result.save("/mnt/user-data/outputs/sticker_demo_plain.jpg")
        print(f"[✓] Saved: sticker_demo_plain.jpg")
        
    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)
    
    print("=" * 70)
