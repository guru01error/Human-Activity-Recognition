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


                          ## matrix graph plot
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Confusion Matrix Calculate Karein
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Labels set karein (apne dataset CSVs ke according)
labels = ["sitting", "standing", "walking", "waving"]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Chart Format aur Style
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(cmap="Blues", ax=ax, values_format="d")
plt.title("Confusion Matrix - Human Activity Recognition", fontsize=12)

# Outputs folder mai save karein
output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "confusion_matrix.png")

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Chart saved successfully at: {output_path}")