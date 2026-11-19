
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_engine.services.fraud_service import FraudService

TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

# Load a real transaction
df = pd.read_csv(TEST_PATH)

row = df.iloc[0]

transaction = row.drop("Class").to_dict()

# Create fraud service
service = FraudService()

print("\n--- ANALYSING AND SAVING TRANSACTION ---")

result = service.analyze_and_save_transaction(
    transaction=transaction,
    transaction_reference="TXN_INTEGRATION_001",
    amount=float(transaction["Amount"])
)

print("\n--- SAVED TRANSACTION RESULT ---")

for key, value in result.items():
    print(f"{key}: {value}")

print("\n--- INTEGRATION TEST COMPLETED ---")
