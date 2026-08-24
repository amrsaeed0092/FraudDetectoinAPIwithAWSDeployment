"""Tests for fraud scoring endpoints."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


VALID_TRANSACTION = {
    "transaction_id": "txn_00000001",
    "customer_token": "customer_token_001",
    "merchant_id": "merchant_electronics_01",
    "amount": 1500.00,
    "currency": "USD",
    "transaction_time": "2026-08-22T18:00:00Z",
    "merchant_category": "electronics",
    "country": "US",
    "is_new_device": True,
    "transactions_last_hour": 9,
}

#test for the health get endpoint
def test_health_endpoint_returns_ok() -> None:
    """The service should expose a healthy status."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

#test for the fraud scoring page or endpoint
def test_high_risk_transaction_is_blocked() -> None:
    """High-risk patterns should result in a BLOCK decision."""
    response = client.post(
        "/api/v1/fraud/score",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 200

    body = response.json()
    assert body["transaction_id"] == "txn_00000001"
    assert body["risk_score"] == 0.85
    assert body["decision"] == "BLOCK"
    assert body["model_version"] == "rule-based-v1"
    assert "HIGH_TRANSACTION_AMOUNT" in body["reason_codes"]
    assert "NEW_DEVICE" in body["reason_codes"]
    assert "HIGH_TRANSACTION_VELOCITY" in body["reason_codes"]


def test_unknown_sensitive_field_is_rejected() -> None:
    """The API must reject unexpected fields such as a card number."""
    unsafe_request = {
        **VALID_TRANSACTION,
        "card_number": "4111111111111111",
    }

    response = client.post(
        "/api/v1/fraud/score",
        json=unsafe_request,
    )

    assert response.status_code == 422