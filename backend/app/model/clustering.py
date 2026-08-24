"""KMeans clustering for the fraud-model training pipeline."""

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringResult:
    """Output summary from the clustering stage."""

    number_of_clusters: int
    train_rows: int
    test_rows: int
    test_evaluation_metric: str
    cluster_summary: dict[int, dict[str, int]]
    test_cluster_metrics: dict[int, str]


class FraudClusterManager:
    """Splits data, finds K, creates clusters, and saves artifacts."""

    feature_columns = [
        "Time",
        "Amount",
        *[f"V{index}" for index in range(1, 29)],
    ]
    target_column = "Class"

    def __init__(
        self,
        data_path: Path,
        artifacts_directory: Path,
        processed_data_directory: Path,
        reports_directory: Path,
    ) -> None:
        self.data_path = data_path
        self.artifacts_directory = artifacts_directory
        self.processed_data_directory = processed_data_directory
        self.reports_directory = reports_directory

    def create_clusters(self) -> ClusteringResult:
        """Create train/test clusters without leaking test data."""
        dataframe = pd.read_csv(self.data_path)

        features = dataframe[self.feature_columns]
        labels = dataframe[self.target_column]

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.20,
            stratify=labels,
            random_state=42,
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        number_of_clusters = self._find_optimal_cluster_count(
            x_train_scaled
        )

        kmeans = KMeans(
            n_clusters=number_of_clusters,
            init="k-means++",
            n_init=10,
            random_state=42,
        )

        train_clusters = kmeans.fit_predict(x_train_scaled)
        test_clusters = kmeans.predict(x_test_scaled)

        train_with_clusters = self._build_clustered_dataframe(
            features=x_train,
            labels=y_train,
            clusters=train_clusters,
        )

        test_with_clusters = self._build_clustered_dataframe(
            features=x_test,
            labels=y_test,
            clusters=test_clusters,
        )

        test_evaluation_metric = self._get_evaluation_metric(y_test)

        cluster_summary, test_cluster_metrics = self._save_artifacts(
            scaler=scaler,
            kmeans=kmeans,
            train_with_clusters=train_with_clusters,
            test_with_clusters=test_with_clusters,
            number_of_clusters=number_of_clusters,
            test_evaluation_metric=test_evaluation_metric,
        )

        return ClusteringResult(
            number_of_clusters=number_of_clusters,
            train_rows=len(train_with_clusters),
            test_rows=len(test_with_clusters),
            test_evaluation_metric=test_evaluation_metric,
            cluster_summary=cluster_summary,
            test_cluster_metrics=test_cluster_metrics,
        )

    def _find_optimal_cluster_count(self, x_train_scaled) -> int:
        """Use KneeLocator; use silhouette score if no knee is found."""
        k_values = list(range(2, 9))
        inertias: list[float] = []
        silhouette_scores: dict[int, float] = {}

        sample_size = min(10_000, len(x_train_scaled))
        sample = x_train_scaled[:sample_size]

        for k in k_values:
            candidate_model = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                random_state=42,
            )

            candidate_labels = candidate_model.fit_predict(x_train_scaled)

            inertias.append(float(candidate_model.inertia_))

            silhouette_scores[k] = float(
                silhouette_score(
                    sample,
                    candidate_labels[:sample_size],
                )
            )

        self._save_elbow_plot(k_values, inertias)

        knee_locator = KneeLocator(
            k_values,
            inertias,
            curve="convex",
            direction="decreasing",
        )

        if knee_locator.elbow is not None:
            return int(knee_locator.elbow)

        return max(silhouette_scores, key=silhouette_scores.get)

    def _build_clustered_dataframe(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        clusters,
    ) -> pd.DataFrame:
        """Attach the original label and the assigned cluster."""
        clustered_data = features.copy()
        clustered_data["Class"] = labels
        clustered_data["Cluster"] = clusters
        return clustered_data

    @staticmethod
    def _get_evaluation_metric(test_y: pd.Series) -> str:
        """Select a safe metric based on available test classes.

        ROC-AUC requires both Class=0 and Class=1. If the test set
        contains only one class, model selection must use accuracy.
        """
        if test_y.empty:
            return "not_available"

        if len(test_y.unique()) == 1:
            return "accuracy"

        return "roc_auc"

    def _save_artifacts(
        self,
        scaler: StandardScaler,
        kmeans: KMeans,
        train_with_clusters: pd.DataFrame,
        test_with_clusters: pd.DataFrame,
        number_of_clusters: int,
        test_evaluation_metric: str,
    ) -> tuple[dict[int, dict[str, int]], dict[int, str]]:
        """Save reusable clustering artifacts and per-cluster datasets."""
        self.artifacts_directory.mkdir(parents=True, exist_ok=True)
        self.processed_data_directory.mkdir(parents=True, exist_ok=True)
        self.reports_directory.mkdir(parents=True, exist_ok=True)

        clusters_directory = self.processed_data_directory / "clusters"
        clusters_directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            scaler,
            self.artifacts_directory / "feature_scaler.joblib",
        )

        joblib.dump(
            kmeans,
            self.artifacts_directory / "kmeans_clusterer.joblib",
        )

        train_with_clusters.to_csv(
            self.processed_data_directory / "train_with_clusters.csv",
            index=False,
        )

        test_with_clusters.to_csv(
            self.processed_data_directory / "test_with_clusters.csv",
            index=False,
        )

        cluster_summary: dict[int, dict[str, int]] = {}
        test_cluster_metrics: dict[int, str] = {}

        for cluster_id in range(number_of_clusters):
            cluster_train_data = train_with_clusters[
                train_with_clusters["Cluster"] == cluster_id
            ]

            cluster_test_data = test_with_clusters[
                test_with_clusters["Cluster"] == cluster_id
            ]

            cluster_train_data.to_csv(
                clusters_directory / f"cluster_{cluster_id}_train.csv",
                index=False,
            )

            cluster_test_data.to_csv(
                clusters_directory / f"cluster_{cluster_id}_test.csv",
                index=False,
            )

            test_metric = self._get_evaluation_metric(
                cluster_test_data["Class"]
            )

            cluster_summary[cluster_id] = {
                "train_rows": int(len(cluster_train_data)),
                "train_fraud_rows": int(cluster_train_data["Class"].sum()),
                "test_rows": int(len(cluster_test_data)),
                "test_fraud_rows": int(cluster_test_data["Class"].sum()),
            }

            test_cluster_metrics[cluster_id] = test_metric

        metadata = {
            "number_of_clusters": number_of_clusters,
            "feature_columns": self.feature_columns,
            "global_test_evaluation_metric": test_evaluation_metric,
            "cluster_summary": cluster_summary,
            "cluster_test_evaluation_metrics": test_cluster_metrics,
        }

        with open(
            self.artifacts_directory / "cluster_metadata.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2)

        return cluster_summary, test_cluster_metrics

    def _save_elbow_plot(
        self,
        k_values: list[int],
        inertias: list[float],
    ) -> None:
        """Save the KMeans elbow curve."""
        self.reports_directory.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 5))
        plt.plot(k_values, inertias, marker="o")
        plt.title("KMeans Elbow Curve")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("Inertia")
        plt.xticks(k_values)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(self.reports_directory / "kmeans_elbow.png")
        plt.close()