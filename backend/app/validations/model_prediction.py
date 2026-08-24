"""Pydantic schemas for the trained Kaggle fraud model."""

from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class KaggleTransactionRequest(BaseModel):
    """Validated input for the Kaggle demo prediction endpoint."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    transaction_id: str = Field(
        default_factory=lambda: f"demo_{uuid4().hex}",
        min_length=8,
        max_length=80,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    time: float = Field(ge=0)
    amount: float = Field(ge=0)
    v_features: list[float] = Field(
        min_length=28,
        max_length=28,
    )

    def to_feature_dataframe(self) -> pd.DataFrame:
        """Convert the validated request into the model feature schema."""
        feature_data = {
            "Time": self.time,
            "Amount": self.amount,
        }

        for index, value in enumerate(self.v_features, start=1):
            feature_data[f"V{index}"] = value

        return pd.DataFrame([feature_data])


class FraudPredictionResponse(BaseModel):
    """Prediction returned by the trained cluster-routing system."""

    transaction_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    decision: str
    cluster_id: int
    algorithm: str
    model_type: str = "kaggle-clustered-demo"