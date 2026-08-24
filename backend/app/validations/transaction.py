"""Pydantic schemas for transaction scoring."""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionRequest(BaseModel):
    """Validated transaction data accepted by the scoring API."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    transaction_id: str = Field(
        min_length=8,
        max_length=80,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    customer_token: str = Field(
        min_length=8,
        max_length=120,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    merchant_id: str = Field(
        min_length=3,
        max_length=80,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    amount: Decimal = Field(
        gt=0,
        max_digits=12,
        decimal_places=2,
    )
    currency: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
    transaction_time: datetime
    merchant_category: str = Field(min_length=2, max_length=80)
    country: str = Field(
        min_length=2,
        max_length=2,
        pattern=r"^[A-Z]{2}$",
    )
    is_new_device: bool = False
    transactions_last_hour: int = Field(ge=0, le=10_000)

    @field_validator("transaction_time")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Reject timestamps without timezone information."""
        if value.tzinfo is None:
            raise ValueError("transaction_time must include a timezone")
        return value


class FraudScoreResponse(BaseModel):
    """Safe API response returned after scoring a transaction."""

    transaction_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    decision: Literal["APPROVE", "REVIEW", "BLOCK"]
    reason_codes: list[str]
    model_version: str