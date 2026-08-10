# Human Activity Recognition using MediaPipe Pose

A real-time **Human Activity Recognition (HAR)** system that uses computer vision and machine learning to identify human activities through a webcam.

The system extracts human body pose landmarks using **MediaPipe Pose** and classifies the detected activity using a **Random Forest machine learning model**.

## 🚀 Features

* Real-time activity recognition using a webcam
* Human pose detection using MediaPipe
* 33 body landmarks extracted from the human pose
* Machine learning classification using Random Forest
* Prediction confidence displayed in real time
* Prediction stabilization using recent predictions
* Custom OpenCV-based dashboard UI
* Supports multiple human activities

## 🎯 Supported Activities

The current model recognizes four activities:

* 🧍 Standing
* 🪑 Sitting
* 🚶 Walking
* 👋 Waving

## 🛠️ Technologies Used

| Technology     | Purpose                               |
| -------------- | ------------------------------------- |
| Python         | Core programming language             |
| OpenCV         | Computer vision and webcam processing |
| MediaPipe Pose | Human pose landmark detection         |
| NumPy          | Numerical operations                  |
| Pandas         | Dataset processing                    |
| Scikit-learn   | Machine learning                      |
| Random Forest  | Activity classification               |
| Joblib         | Model saving and loading              |

## 🧠 How It Works

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Pose
   ↓
Pose Landmarks
   ↓
Feature Extraction
   ↓
Random Forest Classifier
   ↓
Activity Prediction
   ↓
Dashboard UI
```

The webcam captures the user's movement. MediaPipe Pose detects body landmarks, which are converted into numerical features. These features are passed to the trained Random Forest model to predict the current activity.

## 📂 Project Structure

```text
Human-Activity-Recognition/
│
├── app.py
├── collect_data.py
├── train_model.py
├── requirements.txt
│
├── dataset/
│   ├── standing.csv
│   ├── sitting.csv
│   ├── walking.csv
│   └── waving.csv
│
├── models/
│   └── activity_model.pkl
│
└── src/
    ├── camera.py
    ├── predictor.py
    └── ui.py
```

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Human-Activity-Recognition
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

Start the real-time recognition system:

```bash
python app.py
```

The application will open the webcam and display the detected activity along with the prediction confidence.

## 📊 Machine Learning

The project uses a **Random Forest Classifier** for activity recognition.

### Input

Pose landmark coordinates extracted from MediaPipe:

* X coordinate
* Y coordinate
* Z coordinate

These features are used to represent the body posture and movement.

### Output

The trained model predicts one of the four activity classes:

```text
Standing
Sitting
Walking
Waving
```

## 🔄 Data Collection and Training

New training data can be collected using:

```bash
python collect_data.py
```

After collecting the required data, train the model using:

```bash
python train_model.py
```

The trained model is saved inside:

```text
models/activity_model.pkl
```

## 🖥️ Real-Time Dashboard

The project includes a custom OpenCV dashboard that displays:

* Live camera feed
* Detected activity
* Prediction confidence
* Real-time recognition status

## 📌 Future Improvements

* Add more human activities
* Improve recognition in different environments
* Support multiple people
* Improve model generalization
* Add activity history and statistics
* Develop a web-based interface
* Deploy the system as a complete application

## 👨‍💻 Project

**Human Activity Recognition using MediaPipe Pose and Random Forest**

This project demonstrates the integration of **Computer Vision, Pose Estimation, Feature Engineering, and Machine Learning** for real-time human activity classification.

---

⭐ If you find this project useful, consider giving the repository a star.
