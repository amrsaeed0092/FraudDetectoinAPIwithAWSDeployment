"""Command-line entry point for clustered fraud-model training."""

from pathlib import Path

from app.model.cluster_model_trainer import ClusterModelTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    """Train cluster models and save the best model for each cluster."""
    trainer = ClusterModelTrainer(
        processed_data_directory=PROJECT_ROOT / "data" / "processed",
        artifacts_directory=PROJECT_ROOT / "model" / "artifacts",
        reports_directory=PROJECT_ROOT / "data" / "reports",
        tracking_directory=PROJECT_ROOT / "mlruns",
    )

    results = trainer.train_all_clusters()

    print("Cluster model training completed successfully.\n")

    for result in results:
        print(
            f"Cluster {result.cluster_id}: "
            f"{result.algorithm_name} | "
            f"CV AUC={result.selection_auc_cv:.4f} | "
            f"Test {result.test_metric}={result.test_score:.4f} | "
            f"Model={result.model_path}"
        )


if __name__ == "__main__":
    main()