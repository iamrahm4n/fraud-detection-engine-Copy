from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"

print("\n--- LOADING FINAL MODEL ---")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]

print("Model:", bundle["model_name"])
print("Threshold:", threshold)

print("\n--- LOADING TEST DATA ---")

df = pd.read_csv(TEST_PATH)

X_test = df[features]
y_test = df["Class"]

print("Test dataset:", X_test.shape)
print("Actual fraud cases:", int(y_test.sum()))

print("\n--- EVALUATING MODEL ---")

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

average_precision = average_precision_score(y_test, probabilities)

print("Average Precision:", round(average_precision, 4))

print("\n--- FINAL EVALUATION COMPLETED ---")
