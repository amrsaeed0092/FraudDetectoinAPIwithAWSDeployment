"""API routes for trained fraud-model prediction."""

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.model.prediction_service import (
    ModelNotReadyError,
    PredictionService,
)
from app.validations.model_prediction import (
    FraudPredictionResponse,
    KaggleTransactionRequest,
)

router = APIRouter(
    prefix="/demo/fraud",
    tags=["PredictionService"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@lru_cache
def get_prediction_service() -> PredictionService:
    """Create one reusable prediction service per API process."""
    return PredictionService(
        artifacts_directory=PROJECT_ROOT / "model" / "artifacts",
        decision_threshold=0.80,
    )


@router.post(
    "/score",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
)
def score_trained_model(
    transaction: KaggleTransactionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> FraudPredictionResponse:
    """Route a transaction to its cluster model and return fraud risk."""
    try:
        result = service.predict(transaction)
    except ModelNotReadyError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    return FraudPredictionResponse(
        transaction_id=transaction.transaction_id,
        risk_score=result.risk_score,
        decision=result.decision,
        cluster_id=result.cluster_id,
        algorithm=result.algorithm,
    )