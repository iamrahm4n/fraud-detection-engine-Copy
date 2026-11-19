
from pathlib import Path
import joblib
import pandas as pd

from fraud_engine.detection.risk_engine import RiskEngine
from fraud_engine.database.database import SessionLocal
from fraud_engine.database.repository import save_transaction


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"


class FraudService:

    def __init__(self):

        # Load trained machine learning model
        model_bundle = joblib.load(MODEL_PATH)

        self.model = model_bundle["model"]
        self.model.n_jobs = 1

        self.features = model_bundle["features"]

        # Configure risk engine
        self.risk_engine = RiskEngine(
            review_threshold=0.20,
            decline_threshold=0.50
        )

    def analyze_transaction(self, transaction):

        # Validate transaction input
        missing_features = set(self.features) - set(transaction.keys())

        if missing_features:
            raise ValueError(
                f"Missing transaction features: {sorted(missing_features)}"
            )

        # Convert transaction into DataFrame
        transaction_df = pd.DataFrame(
            [[transaction[feature] for feature in self.features]],
            columns=self.features
        )

        # Predict fraud probability
        fraud_probability = float(
            self.model.predict_proba(transaction_df)[0][1]
        )

        # Assess risk
        result = self.risk_engine.assess_transaction(
            fraud_probability
        )

        return result

    def analyze_and_save_transaction(
        self,
        transaction,
        transaction_reference,
        amount
    ):

        # Analyse transaction
        result = self.analyze_transaction(transaction)

        # Open database session
        db = SessionLocal()

        try:
            # Save assessment
            saved_transaction = save_transaction(
                db=db,
                transaction_reference=transaction_reference,
                amount=amount,
                fraud_probability=result["fraud_probability"],
                risk_level=result["risk_level"],
                decision=result["decision"]
            )

            result["transaction_id"] = saved_transaction.id
            result["transaction_reference"] = transaction_reference
            result["amount"] = amount

            return result

        finally:
            db.close()
