
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score
)


# ==================================================
# 1. LOCATE PROJECT FILES
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" / "baseline_model.joblib"
)


# ==================================================
# 2. LOAD TEST DATA AND SAVED MODEL
# ==================================================

print("\n--- LOADING TEST DATA ---")

test_data = pd.read_csv(TEST_PATH)

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

print("Test dataset:", X_test.shape)
print("Actual fraud cases:", y_test.sum())

print("\n--- LOADING TRAINED MODEL ---")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# ==================================================
# 3. CALCULATE FRAUD PROBABILITIES
# ==================================================

print("\n--- GENERATING PREDICTIONS ---")

fraud_probabilities = model.predict_proba(
    X_test
)[:, 1]


# ==================================================
# 4. APPLY SELECTED THRESHOLD
# ==================================================

THRESHOLD = 0.90

y_pred = (
    fraud_probabilities >= THRESHOLD
).astype(int)

print("Selected threshold:", THRESHOLD)


# ==================================================
# 5. EVALUATE MODEL
# ==================================================

print("\n--- TEST CONFUSION MATRIX ---")

print(confusion_matrix(y_test, y_pred))


print("\n--- TEST CLASSIFICATION REPORT ---")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("\n--- TEST AVERAGE PRECISION ---")

print(
    average_precision_score(
        y_test,
        fraud_probabilities
    )
)


# ==================================================
# 6. COMPLETION
# ==================================================

print("\n--- FINAL TEST EVALUATION COMPLETED ---")
