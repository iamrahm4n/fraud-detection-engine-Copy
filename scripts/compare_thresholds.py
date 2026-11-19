from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. LOAD DATASET
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=["Class"])
y = df["Class"]


# ==================================================
# 2. CREATE VALIDATION DATA
# ==================================================

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# 3. DEFINE MODELS
# ==================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ==================================================
# 4. DEFINE THRESHOLDS
# ==================================================

thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]


# ==================================================
# 5. TRAIN AND COMPARE
# ==================================================

results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1]
        ).ravel()

        results.append({
            "Model": model_name,
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "True Positives": tp,
            "False Positives": fp,
            "False Negatives": fn
        })


# ==================================================
# 6. DISPLAY RESULTS
# ==================================================

results_df = pd.DataFrame(results)

print("\n--- COMPLETE THRESHOLD COMPARISON ---")

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ==================================================
# 7. SAVE RESULTS
# ==================================================

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "threshold_comparison.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:", OUTPUT_PATH)

print("\nThreshold comparison completed!")
