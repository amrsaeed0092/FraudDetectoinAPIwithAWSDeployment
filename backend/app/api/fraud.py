"""Fraud-scoring API routes."""

from fastapi import APIRouter, status

from app.model.scoring_service import FraudScoringService
from app.validations.transaction import FraudScoreResponse, TransactionRequest

router = APIRouter(prefix="/fraud", tags=["fraud"])

scoring_service = FraudScoringService()


@router.post(
    "/score",
    response_model=FraudScoreResponse,
    status_code=status.HTTP_200_OK,
)
def score_transaction(transaction: TransactionRequest) -> FraudScoreResponse:
    """Validate and score one transaction."""
    return scoring_service.score(transaction)