import os
import joblib
import numpy as np

class ActivityPredictor:

    def __init__(self, model_path="models/activity_model.pkl"):
        """Loads the pre-trained Scikit-Learn activity classification model."""
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Safely loads the trained model file with path validation."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"[INFO] Successfully loaded model from '{self.model_path}'")
            except Exception as e:
                print(f"[ERROR] Failed to load model file: {e}")
        else:
            print(f"[ERROR] Model file not found at path: '{self.model_path}'")

    def predict(self, landmarks):
        """
        Predicts activity class and confidence percentage based on pose landmarks.
        """
        if self.model is None:
            return "NO MODEL", 0.0

        if not landmarks or len(landmarks) == 0:
            return "NO POSE DETECTED", 0.0

        try:
            # Ensure input features are in 2D array format
            features = np.array(landmarks).reshape(1, -1)

            # Predict Activity Class
            prediction = self.model.predict(features)[0]

            # Predict Confidence Score (Percentage)
            confidence = 0.0
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(max(probabilities) * 100)

            return str(prediction), round(confidence, 1)

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return "ERROR", 0.0