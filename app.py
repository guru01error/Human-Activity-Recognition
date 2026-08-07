import cv2
import mediapipe as mp
from collections import Counter, deque

from src.camera import Camera
from src.predictor import ActivityPredictor
from src.ui import draw_ui

# Load Predictor
predictor = ActivityPredictor()

# Store Last 5 Predictions
prediction_history = deque(maxlen=10)

# MediaPipe Pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Camera
camera = Camera()

cv2.namedWindow("AI Human Activity Recognition System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Human Activity Recognition System", 1200, 700)

while True:

    ret, frame = camera.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:

        # Draw Skeleton
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Extract Landmarks
        landmarks = []

        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Predict Activity
        activity, confidence = predictor.predict(landmarks)

        # Save Prediction
        prediction_history.append(activity)

        # Majority Voting
        stable_activity = Counter(prediction_history).most_common(1)[0][0]

        # Draw Dashboard
        dashboard = draw_ui(frame, stable_activity, confidence)

    else:
      dashboard = draw_ui(frame, stable_activity, confidence)

    cv2.imshow("AI Human Activity Recognition System", dashboard)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()