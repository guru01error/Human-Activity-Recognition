import cv2
import mediapipe as mp
import csv
import os

# Available Activities
activities = ["standing", "sitting", "walking", "waving"]

# User Input
activity = input(f"Enter Activity {activities}: ").strip().lower()

if activity not in activities:
    print("Invalid Activity!")
    exit()

# Create Dataset Folder
os.makedirs("dataset", exist_ok=True)

csv_file = os.path.join("dataset", f"{activity}.csv")

# MediaPipe Pose 
mp_pose = mp.solutions.pose # type: ignore
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils # type: ignore

# Open Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

sample_count = 0

with open(csv_file, "a", newline="") as file:
    writer = csv.writer(file)

    while True:

        ret, frame = cap.read()

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

            # Extract x, y, z Coordinates
            row = []

            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            writer.writerow(row)
            sample_count += 1

        # Display Information
        cv2.putText(frame,
                    f"Activity : {activity}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        cv2.putText(frame,
                    f"Samples : {sample_count}",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2)

        cv2.putText(frame,
                    "Press Q to Stop",
                    (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,255),
                    2)

        cv2.imshow("Human Activity Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("=" * 40)
print(f"Activity : {activity}")
print(f"Total Samples Collected : {sample_count}")
print("=" * 40)