"""Command-line entry point for KMeans clustering."""

from pathlib import Path

from app.model.clustering import FraudClusterManager


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """Create clusters and save reusable clustering artifacts."""
    manager = FraudClusterManager(
        data_path=PROJECT_ROOT / "data" / "processed" / "creditcard_clean.csv",
        artifacts_directory=PROJECT_ROOT / "model" / "artifacts",
        processed_data_directory=PROJECT_ROOT / "data" / "processed",
        reports_directory=PROJECT_ROOT / "data" / "reports",
    )

    result = manager.create_clusters()

    print("Clustering completed successfully.")
    print(f"Selected number of clusters: {result.number_of_clusters}")
    print(f"Training rows: {result.train_rows}")
    print(f"Test rows: {result.test_rows}")
    print(
        "Global test evaluation metric: "
        f"{result.test_evaluation_metric}"
    )

    print("\nCluster summary:")
    for cluster_id, summary in result.cluster_summary.items():
        print(
            f"Cluster {cluster_id}: "
            f"train rows={summary['train_rows']}, "
            f"train fraud rows={summary['train_fraud_rows']}, "
            f"test rows={summary['test_rows']}, "
            f"test fraud rows={summary['test_fraud_rows']}, "
            f"metric={result.test_cluster_metrics[cluster_id]}"
        )


if __name__ == "__main__":
    main()