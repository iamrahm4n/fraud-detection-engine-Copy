from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime
)

from sqlalchemy.orm import declarative_base
from datetime import datetime


# Create the database model base
Base = declarative_base()


class Transaction(Base):

    __tablename__ = "transactions"

    # Unique transaction ID
    id = Column(Integer, primary_key=True, index=True)

    # Transaction information
    transaction_reference = Column(String, unique=True, nullable=False)
    amount = Column(Float, nullable=False)

    # Machine learning prediction
    fraud_probability = Column(Float, nullable=False)

    # Risk engine assessment
    risk_level = Column(String, nullable=False)
    decision = Column(String, nullable=False)

    # Record creation time
    created_at = Column(DateTime, default=datetime.utcnow)
