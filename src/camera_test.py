import cv2
import time
import mediapipe as mp

# MediaPipe Pose Setup
mp_pose = mp.solutions.pose # type: ignore
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils # type: ignore

# Custom Styling for Skeleton Points (LinkedIn Aesthetic)
landmark_style = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3)     # Red Joints
connection_style = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)                 # Cyan Lines

# Open Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera hardware not detected!")
    exit()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture video frame!")
        break

    # Convert BGR to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Draw Pose Skeleton with Custom Colors
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )

    # Calculate Live FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Clean Top Info Header
    cv2.putText(
        frame,
        f"CAMERA TEST | FPS: {int(fps)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        "Press 'Q' to exit test feed",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    cv2.imshow("HAR - Camera Test Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()