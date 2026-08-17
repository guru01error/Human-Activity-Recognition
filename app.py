import cv2
import time
import mediapipe as mp
from collections import Counter, deque

from src.camera import Camera
from src.predictor import ActivityPredictor
from src.ui import draw_ui


def main():
    # Load ML Predictor Engine
    predictor = ActivityPredictor()

    # Temporal Smoothing Queue
    prediction_history = deque(maxlen=10)

    # Initial Fallback State
    stable_activity = "INITIALIZING..."
    confidence = 0.0

    # MediaPipe Pose Setup
    mp_pose = mp.solutions.pose# type: ignore
    mp_draw = mp.solutions.drawing_utils# type: ignore

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # Styling Skeleton
    landmark_spec = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3)
    connection_spec = mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)

    # Camera Stream
    camera = Camera(camera_index=0)

    if not camera.is_opened():
        print("[CRITICAL] Could not access webcam. Exiting application.")
        return

    # Window Setup
    window_name = "AI Human Activity Recognition System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # FPS Variables
    prev_time = time.time()
    fps = 30

    while True:
        ret, frame = camera.read()

        if not ret or frame is None:
            break

        # Calculate FPS Realtime
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 30
        prev_time = curr_time

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Render Skeleton Overlay
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )

            # Flatten Landmark Points
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Model Inference
            raw_activity, confidence = predictor.predict(landmarks)
            prediction_history.append(raw_activity)

            # Majority Vote Activity
            stable_activity = Counter(prediction_history).most_common(1)[0][0]

        else:
            prediction_history.clear()
            stable_activity = "NO PERSON DETECTED"
            confidence = 0.0

        # Render Side-Panel Dashboard (with Live FPS)
        dashboard = draw_ui(
            frame,
            activity=stable_activity,
            confidence=confidence,
            fps=fps
        )

        cv2.imshow(window_name, dashboard)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    pose.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()