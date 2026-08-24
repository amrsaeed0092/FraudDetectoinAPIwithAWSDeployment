"""Temporary rule-based fraud scoring service."""

from app.validations.transaction import FraudScoreResponse, TransactionRequest


class FraudScoringService:
    """Scores transactions.

    This rules-based implementation is temporary. In a later milestone,
    it will load and use the trained scikit-learn model.
    """

    model_version = "rule-based-v1"

    def score(self, transaction: TransactionRequest) -> FraudScoreResponse:
        """Calculate a deterministic fraud-risk score."""
        score = 0.05
        reasons: list[str] = []

        if transaction.amount >= 1_000:
            score += 0.35
            reasons.append("HIGH_TRANSACTION_AMOUNT")

        if transaction.country != "US":
            score += 0.20
            reasons.append("NON_DOMESTIC_TRANSACTION")

        if transaction.is_new_device:
            score += 0.20
            reasons.append("NEW_DEVICE")

        if transaction.transactions_last_hour >= 8:
            score += 0.25
            reasons.append("HIGH_TRANSACTION_VELOCITY")

        risk_score = min(round(score, 2), 1.0)

        if risk_score >= 0.70:
            decision = "BLOCK"
        elif risk_score >= 0.30:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        if not reasons:
            reasons.append("LOW_RISK_PATTERN")

        return FraudScoreResponse(
            transaction_id=transaction.transaction_id,
            risk_score=risk_score,
            decision=decision,
            reason_codes=reasons,
            model_version=self.model_version,
        )