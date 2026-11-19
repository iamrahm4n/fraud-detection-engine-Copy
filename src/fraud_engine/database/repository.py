from sqlalchemy.orm import Session

from fraud_engine.database.models import Transaction


def save_transaction(
    db: Session,
    transaction_reference: str,
    amount: float,
    fraud_probability: float,
    risk_level: str,
    decision: str
):
    # Create a new transaction record
    new_transaction = Transaction(
        transaction_reference=transaction_reference,
        amount=amount,
        fraud_probability=fraud_probability,
        risk_level=risk_level,
        decision=decision
    )

    # Add the record to the database
    db.add(new_transaction)

    # Commit saves the record permanently
    db.commit()

    # Refresh retrieves the saved database values
    db.refresh(new_transaction)

    return new_transaction


def get_transaction(db: Session, transaction_id: int):
    return db.query(Transaction).filter(
        Transaction.id == transaction_id
    ).first()


def get_all_transactions(db: Session):
    return db.query(Transaction).order_by(
        Transaction.id.desc()
    ).all()
