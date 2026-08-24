"""Hyperparameter tuning and best-model selection for fraud clusters."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier



@dataclass
class ModelResult:
    """Result for the selected model in one cluster."""

    cluster_id: int
    algorithm_name: str
    selection_auc_cv: float
    test_score: float
    test_metric: str
    test_pr_auc: float | None
    model_path: str


class ClusterModelTrainer:
    """Tunes five algorithms and selects the strongest model per cluster."""

    feature_columns = [
        "Time",
        "Amount",
        *[f"V{index}" for index in range(1, 29)],
    ]
    target_column = "Class"

    # A cluster needs enough examples from both classes for 3-fold CV.
    minimum_cluster_rows = 200
    minimum_class_rows = 20
    cv_folds = 3

    # Start with 4 trials to validate the complete pipeline.
    search_iterations = 10

    def __init__(
        self,
        processed_data_directory: Path,
        artifacts_directory: Path,
        reports_directory: Path,
        tracking_directory: Path,
    ) -> None:
        self.processed_data_directory = processed_data_directory
        self.artifacts_directory = artifacts_directory
        self.reports_directory = reports_directory
        self.tracking_directory = tracking_directory

    def train_all_clusters(self) -> list[ModelResult]:
        """Train valid cluster models and a global fallback model."""
        self.artifacts_directory.mkdir(parents=True, exist_ok=True)
        self.reports_directory.mkdir(parents=True, exist_ok=True)
        self.tracking_directory.mkdir(parents=True, exist_ok=True)

        database_path = (
            self.tracking_directory / "mlflow.db"
        ).resolve()

        database_uri = f"sqlite:///{database_path.as_posix()}"

        mlflow.set_tracking_uri(database_uri)
        mlflow.set_experiment("fraud-cluster-model-selection")

        train_data = pd.read_csv(
            self.processed_data_directory / "train_with_clusters.csv"
        )
        test_data = pd.read_csv(
            self.processed_data_directory / "test_with_clusters.csv"
        )

        fallback_path = self._train_global_fallback(train_data)

        results: list[ModelResult] = []
        fallback_clusters: list[int] = []

        cluster_ids = sorted(train_data["Cluster"].unique())

        for cluster_id in cluster_ids:
            train_cluster = train_data[train_data["Cluster"] == cluster_id]
            test_cluster = test_data[test_data["Cluster"] == cluster_id]

            if not self._is_cluster_trainable(train_cluster):
                fallback_clusters.append(int(cluster_id))
                continue

            result = self._train_one_cluster(
                cluster_id=int(cluster_id),
                train_cluster=train_cluster,
                test_cluster=test_cluster,
            )
            results.append(result)

        self._save_metadata(
            results=results,
            fallback_clusters=fallback_clusters,
            fallback_model_path=fallback_path,
        )

        return results

    def _is_cluster_trainable(self, cluster_data: pd.DataFrame) -> bool:
        """Require enough rows and enough examples from both classes."""
        if len(cluster_data) < self.minimum_cluster_rows:
            return False

        class_counts = cluster_data[self.target_column].value_counts()

        if len(class_counts) < 2:
            return False

        return int(class_counts.min()) >= self.minimum_class_rows

    def _train_global_fallback(self, train_data: pd.DataFrame) -> str:
        """Train a safe fallback model for sparse clusters."""
        x_train = train_data[self.feature_columns]
        y_train = train_data[self.target_column]

        fallback_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=42,
                    ),
                ),
            ]
        )

        fallback_model.fit(x_train, y_train)

        fallback_path = (
            self.artifacts_directory
            / "fallback_global_logistic_regression.joblib"
        )

        joblib.dump(fallback_model, fallback_path)

        return str(fallback_path)

    def _train_one_cluster(
        self,
        cluster_id: int,
        train_cluster: pd.DataFrame,
        test_cluster: pd.DataFrame,
    ) -> ModelResult:
        """Tune five algorithms, then evaluate the selected one once."""
        x_train = train_cluster[self.feature_columns]
        y_train = train_cluster[self.target_column]

        x_test = test_cluster[self.feature_columns]
        y_test = test_cluster[self.target_column]

        candidates = self._get_model_candidates(y_train)

        best_algorithm_name = ""
        best_model = None
        best_cv_auc = float("-inf")

        for algorithm_name, candidate in candidates.items():
            search = RandomizedSearchCV(
                estimator=candidate["pipeline"],
                param_distributions=candidate["parameters"],
                n_iter=self.search_iterations,
                scoring="roc_auc",
                cv=StratifiedKFold(
                    n_splits=self.cv_folds,
                    shuffle=True,
                    random_state=42,
                ),
                n_jobs=-1,
                random_state=42,
                refit=True,
                error_score="raise",
            )

            fit_parameters = candidate["fit_parameters"]

            search.fit(
                x_train,
                y_train,
                **fit_parameters,
            )

            if search.best_score_ > best_cv_auc:
                best_cv_auc = float(search.best_score_)
                best_algorithm_name = algorithm_name
                best_model = search.best_estimator_

        if best_model is None:
            raise RuntimeError(
                f"No model was selected for cluster {cluster_id}."
            )

        test_score, test_metric, test_pr_auc = self._evaluate_model(
            model=best_model,
            x_test=x_test,
            y_test=y_test,
        )

        model_filename = f"{best_algorithm_name}_{cluster_id}.joblib"
        model_path = self.artifacts_directory / model_filename

        joblib.dump(best_model, model_path)

        with mlflow.start_run(run_name=f"cluster_{cluster_id}_{best_algorithm_name}"):
            mlflow.log_params(
                {
                    "cluster_id": cluster_id,
                    "algorithm": best_algorithm_name,
                    "train_rows": len(train_cluster),
                    "train_fraud_rows": int(y_train.sum()),
                    "test_rows": len(test_cluster),
                    "test_fraud_rows": int(y_test.sum()),
                }
            )

            mlflow.log_metrics(
                {
                    "selection_auc_cv": best_cv_auc,
                    "test_score": test_score,
                }
            )

            if test_pr_auc is not None:
                mlflow.log_metric("test_pr_auc", test_pr_auc)

            if best_algorithm_name == "xgboost":
                mlflow.xgboost.log_model(
                    xgb_model=best_model.named_steps["classifier"],
                    name="model",
                    model_format="json",
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    name="model",
                )

        return ModelResult(
            cluster_id=cluster_id,
            algorithm_name=best_algorithm_name,
            selection_auc_cv=best_cv_auc,
            test_score=test_score,
            test_metric=test_metric,
            test_pr_auc=test_pr_auc,
            model_path=str(model_path),
        )

    def _get_model_candidates(self, y_train: pd.Series) -> dict:
        """Return five tuned model candidates."""
        positive_rows = int(y_train.sum())
        negative_rows = int(len(y_train) - positive_rows)

        scale_pos_weight = negative_rows / max(positive_rows, 1)

        balanced_sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        return {
            "logistic_regression": {
                "pipeline": Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                class_weight="balanced",
                                max_iter=2_000,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                "parameters": {
                    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "classifier__solver": ["liblinear", "lbfgs"],
                },
                "fit_parameters": {},
            },
            "random_forest": {
                "pipeline": Pipeline(
                    steps=[
                        (
                            "classifier",
                            RandomForestClassifier(
                                class_weight="balanced_subsample",
                                n_jobs=-1,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                "parameters": {
                    "classifier__n_estimators": [150, 250],
                    "classifier__max_depth": [None, 10, 20],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__min_samples_leaf": [1, 2, 4],
                },
                "fit_parameters": {},
            },
            "extra_trees": {
                "pipeline": Pipeline(
                    steps=[
                        (
                            "classifier",
                            ExtraTreesClassifier(
                                class_weight="balanced",
                                n_jobs=-1,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                "parameters": {
                    "classifier__n_estimators": [150, 250],
                    "classifier__max_depth": [None, 10, 20],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__min_samples_leaf": [1, 2, 4],
                },
                "fit_parameters": {},
            },
            "hist_gradient_boosting": {
                "pipeline": Pipeline(
                    steps=[
                        (
                            "classifier",
                            HistGradientBoostingClassifier(
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                "parameters": {
                    "classifier__learning_rate": [0.03, 0.05, 0.10],
                    "classifier__max_iter": [100, 200, 300],
                    "classifier__max_leaf_nodes": [15, 31, 63],
                    "classifier__l2_regularization": [0.0, 0.1, 1.0],
                },
                "fit_parameters": {
                    "classifier__sample_weight": balanced_sample_weights,
                },
            },
            "xgboost": {
                "pipeline": Pipeline(
                    steps=[
                        (
                            "classifier",
                            XGBClassifier(
                                objective="binary:logistic",
                                eval_metric="logloss",
                                tree_method="hist",
                                n_jobs=-1,
                                random_state=42,
                                scale_pos_weight=scale_pos_weight,
                            ),
                        ),
                    ]
                ),
                "parameters": {
                    "classifier__n_estimators": [100, 200, 300],
                    "classifier__max_depth": [3, 5, 7],
                    "classifier__learning_rate": [0.03, 0.05, 0.10],
                    "classifier__subsample": [0.7, 0.85, 1.0],
                    "classifier__colsample_bytree": [0.7, 0.85, 1.0],
                },
                "fit_parameters": {},
            },
        }

    @staticmethod
    def _evaluate_model(
        model,
        x_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> tuple[float, str, float | None]:
        """Use ROC-AUC when possible; otherwise use accuracy safely."""
        predictions = model.predict(x_test)

        # ROC-AUC requires two classes in y_test.
        if len(y_test.unique()) == 1:
            return (
                float(accuracy_score(y_test, predictions)),
                "accuracy",
                None,
            )

        probabilities = model.predict_proba(x_test)[:, 1]

        return (
            float(roc_auc_score(y_test, probabilities)),
            "roc_auc",
            float(average_precision_score(y_test, probabilities)),
        )

    def _save_metadata(
        self,
        results: list[ModelResult],
        fallback_clusters: list[int],
        fallback_model_path: str,
    ) -> None:
        """Save routing information required by the prediction API."""
        metadata = {
            "feature_columns": self.feature_columns,
            "fallback_model": fallback_model_path,
            "fallback_clusters": fallback_clusters,
            "cluster_models": {
                str(result.cluster_id): {
                    "algorithm": result.algorithm_name,
                    "selection_auc_cv": result.selection_auc_cv,
                    "test_score": result.test_score,
                    "test_metric": result.test_metric,
                    "test_pr_auc": result.test_pr_auc,
                    "model_path": result.model_path,
                }
                for result in results
            },
        }

        with open(
            self.artifacts_directory / "model_metadata.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2)

        with open(
            self.reports_directory / "cluster_model_results.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                [asdict(result) for result in results],
                file,
                indent=2,
            )