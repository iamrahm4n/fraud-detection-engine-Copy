from pathlib import Path
import time

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. LOCATE AND LOAD DATASET
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

print("\n--- LOADING DATASET ---")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=["Class"])
y = df["Class"]

print("Dataset shape:", df.shape)


# ==================================================
# 2. CREATE VALIDATION SPLIT
# ==================================================

print("\n--- CREATING VALIDATION DATA ---")

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_validation))


# ==================================================
# 3. DEFINE MODELS
# ==================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ==================================================
# 4. TRAIN AND EVALUATE MODELS
# ==================================================

results = []

THRESHOLD = 0.90

for model_name, model in models.items():

    print(f"\n--- TRAINING {model_name.upper()} ---")

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print("Training completed!")
    print(f"Training time: {training_time:.2f} seconds")

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
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

    average_precision = average_precision_score(
        y_validation,
        probabilities
    )

    results.append({
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Average Precision": average_precision,
        "Training Time (seconds)": training_time
    })


# ==================================================
# 5. DISPLAY MODEL COMPARISON
# ==================================================

print("\n--- MODEL COMPARISON RESULTS ---")

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ==================================================
# 6. SAVE COMPARISON RESULTS
# ==================================================

RESULT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "model_comparison.csv"
)

results_df.to_csv(
    RESULT_PATH,
    index=False
)

print("\nComparison results saved to:", RESULT_PATH)

print("\n--- MODEL COMPARISON COMPLETED ---")
