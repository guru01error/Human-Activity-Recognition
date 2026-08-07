import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Dataset Folder
dataset_path = "dataset"

# Activities
activities = [
    "standing",
    "sitting",
    "walking",
    "waving"
]

all_data = []

# Read CSV files
for activity in activities:
    file_path = os.path.join(dataset_path, f"{activity}.csv")

    if not os.path.exists(file_path):
        print(f"{activity}.csv not found")
        continue

    df = pd.read_csv(file_path, header=None)
    df["label"] = activity
    all_data.append(df)

# Check dataset
if len(all_data) == 0:
    print("No dataset found!")
    exit()

# Merge all data
data = pd.concat(all_data, ignore_index=True)

# Features & Labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy * 100:.2f}%")

# Save Model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/activity_model.pkl")

print("Model Saved Successfully!")