class RiskEngine:
    """
    Converts fraud probabilities into risk levels and decisions.
    """

    def __init__(self, review_threshold=0.20, decline_threshold=0.50):
        self.review_threshold = review_threshold
        self.decline_threshold = decline_threshold

        if not 0 <= review_threshold < decline_threshold <= 1:
            raise ValueError("Invalid risk thresholds.")

    def assess_transaction(self, fraud_probability):

        if not 0 <= fraud_probability <= 1:
            raise ValueError("Fraud probability must be between 0 and 1.")

        if fraud_probability >= self.decline_threshold:
            risk_level = "HIGH"
            decision = "DECLINE"

        elif fraud_probability >= self.review_threshold:
            risk_level = "MEDIUM"
            decision = "REVIEW"

        else:
            risk_level = "LOW"
            decision = "APPROVE"

        return {
            "fraud_probability": round(fraud_probability, 4),
            "risk_level": risk_level,
            "decision": decision
        }
