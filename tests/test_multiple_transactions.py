
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

print("1. TEST SCRIPT STARTED", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_engine.services.fraud_service import FraudService

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

print("2. LOADING TEST DATA", flush=True)

df = pd.read_csv(TEST_PATH)

X_test = df.drop(columns=["Class"])
y_actual = df["Class"]

print("3. TEST DATA LOADED:", X_test.shape, flush=True)

print("4. LOADING FRAUD SERVICE", flush=True)

service = FraudService()

print("5. MODEL LOADED", flush=True)

print("6. RUNNING BATCH PREDICTION", flush=True)

probabilities = service.model.predict_proba(
    X_test[service.features]
)[:, 1]

print("7. PREDICTION COMPLETED", flush=True)

y_predicted = (
    probabilities >= service.risk_engine.decline_threshold
).astype(int)

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_actual, y_predicted))

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    y_actual,
    y_predicted,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

print("\n--- TEST COMPLETED ---")
