import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_engine.services.fraud_service import FraudService


# Create FastAPI application
app = FastAPI(
    title="Fraud Detection Engine",
    description="Machine learning powered transaction risk assessment API",
    version="1.0.0"
)

# Load fraud service
fraud_service = FraudService()


# Define incoming transaction structure
class TransactionRequest(BaseModel):

    transaction_reference: str = Field(min_length=1, max_length=100)
    features: dict[str, float]


# Basic health endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Fraud Detection Engine"
    }


# Fraud analysis endpoint
@app.post("/transactions/analyze")
def analyze_transaction(request: TransactionRequest):

    try:
        result = fraud_service.analyze_and_save_transaction(
            transaction=request.features,
            transaction_reference=request.transaction_reference,
            amount=request.features.get("Amount")
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
