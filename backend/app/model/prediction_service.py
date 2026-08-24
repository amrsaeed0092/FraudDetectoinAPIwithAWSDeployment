"""Prediction service for the clustered Kaggle fraud models."""

from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath
from threading import Lock

import joblib

from app.validations.model_prediction import KaggleTransactionRequest


class ModelNotReadyError(RuntimeError):
    """Raised when required model artifacts are missing."""


@dataclass
class PredictionResult:
    """Internal prediction result."""

    risk_score: float
    decision: str
    cluster_id: int
    algorithm: str


class PredictionService:
    """Loads artifacts once and routes transactions to the correct model."""

    def __init__(
        self,
        artifacts_directory: Path,
        decision_threshold: float = 0.50,
    ) -> None:
        self.artifacts_directory = artifacts_directory
        self.decision_threshold = decision_threshold

        self._is_loaded = False
        self._load_lock = Lock()

        self._scaler = None
        self._kmeans = None
        self._fallback_model = None
        self._cluster_models: dict[int, object] = {}
        self._cluster_algorithms: dict[int, str] = {}

    def predict(self, request: KaggleTransactionRequest) -> PredictionResult:
        """Route one validated transaction to the correct trained model."""
        self._load_artifacts_once()

        features = request.to_feature_dataframe()

        scaled_features = self._scaler.transform(features)
        cluster_id = int(self._kmeans.predict(scaled_features)[0])

        model = self._cluster_models.get(cluster_id)
        algorithm = self._cluster_algorithms.get(
            cluster_id,
            "fallback_global_logistic_regression",
        )

        if model is None:
            model = self._fallback_model

        probability = float(model.predict_proba(features)[0, 1])

        decision = (
            "BLOCK"
            if probability >= self.decision_threshold
            else "APPROVE"
        )

        return PredictionResult(
            risk_score=round(probability, 6),
            decision=decision,
            cluster_id=cluster_id,
            algorithm=algorithm,
        )

    def _load_artifacts_once(self) -> None:
        """Load all required artifacts once per API process."""
        if self._is_loaded:
            return

        with self._load_lock:
            if self._is_loaded:
                return

            metadata_path = (
                self.artifacts_directory / "model_metadata.json"
            )

            if not metadata_path.exists():
                raise ModelNotReadyError(
                    f"Missing metadata file: {metadata_path}"
                )

            scaler_path = (
                self.artifacts_directory / "feature_scaler.joblib"
            )
            kmeans_path = (
                self.artifacts_directory / "kmeans_clusterer.joblib"
            )

            if not scaler_path.exists() or not kmeans_path.exists():
                raise ModelNotReadyError(
                    "Missing scaler or KMeans model artifact."
                )

            with open(metadata_path, encoding="utf-8") as file:
                metadata = json.load(file)

            fallback_filename = self._get_filename(
                metadata["fallback_model"]
            )

            fallback_path = (
                self.artifacts_directory / fallback_filename
            )

            if not fallback_path.exists():
                raise ModelNotReadyError(
                    f"Missing fallback model: {fallback_path}"
                )

            self._scaler = joblib.load(scaler_path)
            self._kmeans = joblib.load(kmeans_path)
            self._fallback_model = joblib.load(fallback_path)

            for cluster_id_text, model_info in metadata[
                "cluster_models"
            ].items():
                cluster_id = int(cluster_id_text)

                model_filename = self._get_filename(
                    model_info["model_path"]
                )

                model_path = self.artifacts_directory / model_filename

                if not model_path.exists():
                    raise ModelNotReadyError(
                        f"Missing cluster model: {model_path}"
                    )

                self._cluster_models[cluster_id] = joblib.load(model_path)
                self._cluster_algorithms[cluster_id] = model_info[
                    "algorithm"
                ]

            self._is_loaded = True

    @staticmethod
    def _get_filename(saved_path: str) -> str:
        """Support saved artifact paths from Windows and Linux."""
        return PureWindowsPath(saved_path).name