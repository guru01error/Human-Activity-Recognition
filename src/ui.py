import cv2
import numpy as np

def draw_ui(frame, activity, confidence=None):

    # Frame Size
    height, width = frame.shape[:2]
    panel_width = 220

    # Create Dashboard
    dashboard = np.zeros((height, width + panel_width, 3), dtype=np.uint8)

    # Camera (Left Side)
    dashboard[:, :width] = frame

    # Dashboard Background
    dashboard[:, width:] = (35, 35, 35)

    # Divider Line
    cv2.line(dashboard, (width, 0), (width, height), (120, 120, 120), 2)

    # Title
    cv2.putText(
        dashboard,
        "AI DASHBOARD",
        (width + 15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # Activity
    cv2.putText(
        dashboard,
        "Activity",
        (width + 15, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        2
    )

    cv2.putText(
        dashboard,
        activity,
        (width + 15, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Confidence
    if confidence is not None:

        cv2.putText(
            dashboard,
            "Confidence",
            (width + 15, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            2
        )

        cv2.putText(
            dashboard,
            f"{confidence:.1f}%",
            (width + 15, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Progress Bar Background
        cv2.rectangle(
            dashboard,
            (width + 15, 205),
            (width + 175, 220),
            (80, 80, 80),
            -1
        )

        # Progress Bar
        bar = int((confidence / 100) * 160)

        cv2.rectangle(
            dashboard,
            (width + 15, 205),
            (width + 15 + bar, 220),
            (0, 255, 0),
            -1
        )

    # Exit
    cv2.putText(
        dashboard,
        "Press Q to Exit",
        (width + 15, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    return dashboard