import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_ui(frame, activity, confidence=None):

    # Frame Size
    height, width = frame.shape[:2]
    panel_width = 220

    # Create Dashboard
    dashboard = np.zeros(
        (height, width + panel_width, 3),
        dtype=np.uint8
    )

    # Camera - Left Side
    dashboard[:, :width] = frame

    # Dashboard Background
    dashboard[:, width:] = (35, 35, 35)

    # Divider Line
    cv2.line(
        dashboard,
        (width, 0),
        (width, height),
        (120, 120, 120),
        2
    )

    # Convert OpenCV image to PIL
    pil_image = Image.fromarray(
        cv2.cvtColor(dashboard, cv2.COLOR_BGR2RGB)
    )

    draw = ImageDraw.Draw(pil_image)

    # Times New Roman
    font_path = r"C:\Windows\Fonts\times.ttf"

    title_font = ImageFont.truetype(font_path, 22)
    label_font = ImageFont.truetype(font_path, 18)
    activity_font = ImageFont.truetype(font_path, 24)
    confidence_font = ImageFont.truetype(font_path, 21)
    exit_font = ImageFont.truetype(font_path, 16)

    # Title
    draw.text(
        (width + 15, 12),
        "AI DASHBOARD",
        font=title_font,
        fill=(255, 255, 0)
    )

    # Activity Label
    draw.text(
        (width + 15, 55),
        "Activity",
        font=label_font,
        fill=(180, 180, 180)
    )

    # Activity Value
    draw.text(
        (width + 15, 82),
        str(activity),
        font=activity_font,
        fill=(0, 255, 0)
    )

    # Confidence
    if confidence is not None:

        draw.text(
            (width + 15, 125),
            "Confidence",
            font=label_font,
            fill=(180, 180, 180)
        )

        draw.text(
            (width + 15, 152),
            f"{confidence:.1f}%",
            font=confidence_font,
            fill=(255, 255, 255)
        )

        # Progress Bar Background
        cv2.rectangle(
            dashboard,
            (width + 15, 190),
            (width + 175, 205),
            (80, 80, 80),
            -1
        )

        # Progress Bar
        bar = int(
            max(0, min(float(confidence), 100)) / 100 * 160
        )

        cv2.rectangle(
            dashboard,
            (width + 15, 190),
            (width + 15 + bar, 205),
            (0, 255, 0),
            -1
        )

    # Exit Text
    draw.text(
        (width + 15, height - 32),
        "Press Q to Exit",
        font=exit_font,
        fill=(255, 255, 0)
    )

    # Convert PIL back to OpenCV
    dashboard = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR
    )

    return dashboard