import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def draw_ui(frame, activity, confidence=None, fps=30):
    height, width = frame.shape[:2]
    panel_width = 250  # Slightly wider for better layout fit

    # Dark Theme Base Canvas
    dashboard = np.zeros((height, width + panel_width, 3), dtype=np.uint8)
    dashboard[:, :width] = frame

    # Side Panel Background
    panel_bg_color = (24, 15, 15)  # BGR Dark Navy
    dashboard[:, width:] = panel_bg_color

    # Panel Divider Line
    cv2.line(dashboard, (width, 0), (width, height), (50, 40, 30), 1)

    # PIL Image Setup
    pil_image = Image.fromarray(cv2.cvtColor(dashboard, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Font Setup
    font_path = r"C:\Windows\Fonts\segoeui.ttf"
    if not os.path.exists(font_path):
        font_path = r"C:\Windows\Fonts\arial.ttf"

    try:
        title_font = ImageFont.truetype(font_path, 15)
        label_font = ImageFont.truetype(font_path, 11)
        value_font = ImageFont.truetype(font_path, 20)
        small_font = ImageFont.truetype(font_path, 11)
        stat_font = ImageFont.truetype(font_path, 13)
    except Exception:
        title_font = label_font = value_font = small_font = stat_font = ImageFont.load_default()

    # Dynamic Color Logic
    conf_val = float(confidence) if confidence is not None else 0.0
    if conf_val >= 80:
        accent_color = (16, 185, 129)   # Emerald Green
    elif conf_val >= 50:
        accent_color = (245, 158, 11)   # Amber Yellow
    else:
        accent_color = (239, 68, 68)    # Bright Red

    # 1. LIVE STATUS HEADER
    draw.text((width + 15, 15), "● LIVE AI SYSTEM", font=title_font, fill=(56, 189, 248))

    # 2. ACTIVITY CARD
    draw.rectangle([width + 15, 45, width + panel_width - 15, 110], fill=(30, 41, 59), outline=(51, 65, 85))
    draw.text((width + 25, 53), "DETECTED ACTIVITY", font=label_font, fill=(148, 163, 184))
    act_text = str(activity).upper() if activity else "SCANNING..."
    draw.text((width + 25, 72), act_text, font=value_font, fill=(255, 255, 255))

    # 3. CONFIDENCE CARD
    if confidence is not None:
        draw.rectangle([width + 15, 120, width + panel_width - 15, 200], fill=(30, 41, 59), outline=(51, 65, 85))
        draw.text((width + 25, 128), "MODEL CONFIDENCE", font=label_font, fill=(148, 163, 184))
        draw.text((width + 25, 146), f"{conf_val:.1f}%", font=value_font, fill=accent_color)

        # Progress Bar
        bar_x1, bar_y1 = width + 25, 180
        bar_x2, bar_y2 = width + panel_width - 25, 188
        bar_max_width = bar_x2 - bar_x1
        draw.rectangle([bar_x1, bar_y1, bar_x2, bar_y2], fill=(15, 23, 42))

        fill_width = int(max(0, min(conf_val, 100)) / 100 * bar_max_width)
        if fill_width > 0:
            draw.rectangle([bar_x1, bar_y1, bar_x1 + fill_width, bar_y2], fill=accent_color)

    # 4. NEW: SYSTEM PERFORMANCE METRICS (Khali Space Fill 1)
    draw.rectangle([width + 15, 210, width + panel_width - 15, 290], fill=(30, 41, 59), outline=(51, 65, 85))
    draw.text((width + 25, 218), "SYSTEM METRICS", font=label_font, fill=(148, 163, 184))
    draw.text((width + 25, 238), f"Frame Rate:  {fps} FPS", font=stat_font, fill=(203, 213, 225))
    draw.text((width + 25, 260), f"Resolution:  {width}x{height}", font=stat_font, fill=(203, 213, 225))

    # 5. NEW: PIPELINE & ALGORITHM INFO (Khali Space Fill 2)
    draw.rectangle([width + 15, 300, width + panel_width - 15, 390], fill=(30, 41, 59), outline=(51, 65, 85))
    draw.text((width + 25, 308), "AI PIPELINE INFO", font=label_font, fill=(148, 163, 184))
    draw.text((width + 25, 328), "Pose: MediaPipe 3D", font=stat_font, fill=(148, 163, 184))
    draw.text((width + 25, 348), "Classifier: Random Forest", font=stat_font, fill=(148, 163, 184))
    draw.text((width + 25, 368), "Smoothing: Majority Vote", font=stat_font, fill=(56, 189, 248))

    # FOOTER INSTRUCTIONS
    draw.text((width + 15, height - 25), "Press 'Q' to terminate feed", font=small_font, fill=(100, 116, 139))

    # Convert back to OpenCV
    dashboard = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return dashboard