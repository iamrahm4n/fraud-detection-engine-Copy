from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. LOCATE PROJECT AND DATASET
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "train.csv"
)


# ==================================================
# 2. CHECK DATASET LOCATION
# ==================================================

print("\n--- PROJECT PATH CHECK ---")

print("Project folder:", PROJECT_ROOT)
print("Training dataset:", TRAIN_PATH)
print("Dataset exists:", TRAIN_PATH.exists())

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_PATH}"
    )


# ==================================================
# 3. LOAD TRAINING DATA
# ==================================================

print("\n--- LOADING TRAINING DATA ---")

df = pd.read_csv(TRAIN_PATH)

print("Dataset shape:", df.shape)

X = df.drop(columns=["Class"])
y = df["Class"]

print("Features:", X.shape)
print("Target:", y.shape)


# ==================================================
# 4. CREATE TRAINING AND VALIDATION SETS
# ==================================================

print("\n--- CREATING VALIDATION SPLIT ---")

X_fit, X_validation, y_fit, y_validation = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("Model training rows:", len(X_fit))
print("Validation rows:", len(X_validation))

print("Training fraud cases:", y_fit.sum())
print("Validation fraud cases:", y_validation.sum())


# ==================================================
# 5. BUILD LOGISTIC REGRESSION MODEL
# ==================================================

print("\n--- BUILDING MODEL ---")

model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )
])


# ==================================================
# 6. TRAIN MODEL
# ==================================================

print("\n--- TRAINING MODEL ---")

model.fit(X_fit, y_fit)

print("Model training completed!")


# ==================================================
# 7. CALCULATE FRAUD PROBABILITIES
# ==================================================

print("\n--- CALCULATING FRAUD PROBABILITIES ---")

fraud_probabilities = model.predict_proba(
    X_validation
)[:, 1]

print("Fraud probabilities calculated!")


# ==================================================
# 8. TEST DIFFERENT CLASSIFICATION THRESHOLDS
# ==================================================

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

print("\n--- THRESHOLD COMPARISON ---")

for threshold in thresholds:

    predictions = (
        fraud_probabilities >= threshold
    ).astype(int)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_validation,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1-score: {f1:.4f}"
    )


# ==================================================
# 9. COMPLETION
# ==================================================

print("\n--- THRESHOLD ANALYSIS COMPLETED ---")


# ==================================================
# 10. INSPECT THE 0.90 THRESHOLD
# ==================================================

selected_threshold = 0.90

selected_predictions = (
    fraud_probabilities >= selected_threshold
).astype(int)

print("\n--- CONFUSION MATRIX AT THRESHOLD 0.90 ---")

print(
    confusion_matrix(
        y_validation,
        selected_predictions
    )
)

print("\n--- CLASSIFICATION REPORT AT THRESHOLD 0.90 ---")

print(
    classification_report(
        y_validation,
        selected_predictions,
        zero_division=0
    )
)

print("\nThreshold evaluation completed!")

