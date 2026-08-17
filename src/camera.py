import cv2

class Camera:
    def __init__(self, camera_index=0):
        """Initializes the VideoCapture stream."""
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Unable to open camera source at index {self.camera_index}")

    def read(self):
        """Reads a frame from the webcam feed."""
        if not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def is_opened(self):
        """Checks if the camera stream is currently active."""
        return self.cap.isOpened()

    def release(self):
        """Releases the camera hardware resource safely."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[INFO] Camera hardware resource released successfully.")

    def __del__(self):
        """Garbage collection fallback cleanup."""
        self.release()