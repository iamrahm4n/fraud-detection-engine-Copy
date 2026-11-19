from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"

# Load training data
print("\n--- LOADING TRAINING DATA ---")

df = pd.read_csv(TRAIN_PATH)

X = df.drop(columns=["Class"])
y = df["Class"]

print("Training dataset:", X.shape)
print("Fraud cases:", int(y.sum()))

# Create selected model
print("\n--- TRAINING FINAL RANDOM FOREST MODEL ---")

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

print("Training completed!")

# Save model and threshold together
model_bundle = {
    "model": model,
    "threshold": 0.50,
    "features": list(X.columns),
    "model_name": "Random Forest"
}

joblib.dump(model_bundle, MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)
print("Selected threshold:", model_bundle["threshold"])
print("\nFinal model training completed!")
