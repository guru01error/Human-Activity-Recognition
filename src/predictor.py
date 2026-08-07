import joblib

class ActivityPredictor:

    def __init__(self):
        self.model = joblib.load("models/activity_model.pkl")

    def predict(self, landmarks):

        # Predict Activity
        prediction = self.model.predict([landmarks])[0]

        # Predict Confidence
        confidence = None

        if hasattr(self.model, "predict_proba"):
            confidence = max(self.model.predict_proba([landmarks])[0]) * 100

        return prediction, confidence