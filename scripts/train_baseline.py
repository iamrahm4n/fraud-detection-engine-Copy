
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. LOCATE PROJECT FILES
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.joblib"


# ==================================================
# 2. LOAD TRAINING AND TESTING DATA
# ==================================================

print("\nLoading datasets...")

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

X_train = train_data.drop(columns=["Class"])
y_train = train_data["Class"]

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ==================================================
# 3. BUILD THE BASELINE MODEL
# ==================================================

print("\nBuilding Logistic Regression model...")

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
# 4. TRAIN THE MODEL
# ==================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==================================================
# 5. MAKE PREDICTIONS
# ==================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ==================================================
# 6. EVALUATE MODEL PERFORMANCE
# ==================================================

print("\n--- CONFUSION MATRIX ---")

print(confusion_matrix(y_test, y_pred))

print("\n--- CLASSIFICATION REPORT ---")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\n--- PRECISION-RECALL AVERAGE PRECISION ---")

print(average_precision_score(y_test, y_prob))


# ==================================================
# 7. SAVE TRAINED MODEL
# ==================================================

print("\nSaving trained model...")

joblib.dump(model, MODEL_PATH)

print("Model saved to:", MODEL_PATH)

print("\nBaseline training completed successfully!")
