"""tests for the trained-model API."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_with_clusters.csv"
)

MODEL_METADATA_PATH = (
    PROJECT_ROOT
    / "model"
    / "artifacts"
    / "model_metadata.json"
)

'''
    those three trained-model tests should run locally, 
    but be skipped safely in CI when the dataset/models are unavailable.
'''
if not TEST_DATA_PATH.exists() or not MODEL_METADATA_PATH.exists():
    pytest.skip(
        "Trained-model artifacts are unavailable in this environment.",
        allow_module_level=True,
    )

@pytest.fixture(scope="module")
def valid_prediction_payload() -> tuple[dict, int]:
    """Create one valid API payload from a real held-out test row."""
    dataframe = pd.read_csv(TEST_DATA_PATH)
    row = dataframe.iloc[0]

    payload = {
        "transaction_id": "test_prediction_000001",
        "time": float(row["Time"]),
        "amount": float(row["Amount"]),
        "v_features": [
            float(row[f"V{index}"])
            for index in range(1, 29)
        ],
    }

    expected_cluster = int(row["Cluster"])

    return payload, expected_cluster


def test_trained_model_routes_to_expected_cluster(
    valid_prediction_payload: tuple[dict, int],
) -> None:
    """The API must use the same KMeans cluster as the saved test data."""
    payload, expected_cluster = valid_prediction_payload

    response = client.post(
        "/api/v1/demo/fraud/score",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["transaction_id"] == "test_prediction_000001"
    assert body["cluster_id"] == expected_cluster
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["decision"] in {"APPROVE", "BLOCK"}
    assert body["algorithm"]
    assert body["model_type"] == "kaggle-clustered-demo"


def test_prediction_rejects_incorrect_feature_count() -> None:
    """The API must reject a request without all 28 V features."""
    invalid_payload = {
        "transaction_id": "invalid_feature_count",
        "time": 100.0,
        "amount": 50.0,
        "v_features": [0.0] * 27,
    }

    response = client.post(
        "/api/v1/demo/fraud/score",
        json=invalid_payload,
    )

    assert response.status_code == 422


def test_prediction_rejects_training_label() -> None:
    """The scoring API must reject Class because it is training-only."""
    invalid_payload = {
        "transaction_id": "invalid_training_label",
        "time": 100.0,
        "amount": 50.0,
        "v_features": [0.0] * 28,
        "Class": 1,
    }

    response = client.post(
        "/api/v1/demo/fraud/score",
        json=invalid_payload,
    )

    assert response.status_code == 422
#python -m pytest -q
#python -m pytest --collect-only -q