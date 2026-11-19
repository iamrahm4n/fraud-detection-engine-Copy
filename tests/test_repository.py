import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_engine.database.database import SessionLocal, engine
from fraud_engine.database.models import Base
from fraud_engine.database.repository import (
    save_transaction,
    get_transaction,
    get_all_transactions
)

# Make sure database tables exist
Base.metadata.create_all(bind=engine)

# Open database session
db = SessionLocal()

try:
    print("\n--- SAVING TRANSACTION ---")

    transaction = save_transaction(
        db=db,
        transaction_reference="TXN_TEST_001",
        amount=2500.00,
        fraud_probability=0.85,
        risk_level="HIGH",
        decision="DECLINE"
    )

    print("Transaction saved!")
    print("Database ID:", transaction.id)

    print("\n--- RETRIEVING TRANSACTION ---")

    saved_transaction = get_transaction(db, transaction.id)

    print("Reference:", saved_transaction.transaction_reference)
    print("Amount:", saved_transaction.amount)
    print("Fraud probability:", saved_transaction.fraud_probability)
    print("Risk level:", saved_transaction.risk_level)
    print("Decision:", saved_transaction.decision)

    print("\n--- ALL TRANSACTIONS ---")

    all_transactions = get_all_transactions(db)

    print("Total saved transactions:", len(all_transactions))

finally:
    db.close()

print("\nDATABASE TEST COMPLETED!")
