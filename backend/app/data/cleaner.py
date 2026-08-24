"""Dataset cleaning for credit-card fraud detection."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CleaningSummary:
    """Summary returned after one data-cleaning run."""

    input_rows: int
    output_rows: int
    duplicates_removed: int
    missing_values_removed: int


class FraudDataCleaner:
    """Validates and cleans the raw credit-card fraud dataset."""

    required_columns = {"Time", "Amount", "Class"}

    def __init__(self, input_path: Path, output_path: Path) -> None:
        self.input_path = input_path
        self.output_path = output_path

    def clean(self) -> CleaningSummary:
        """Read, validate, clean, and save the dataset."""
        dataframe = pd.read_csv(self.input_path)

        self._validate_required_columns(dataframe)

        input_rows = len(dataframe)

        dataframe = dataframe.drop_duplicates()
        duplicates_removed = input_rows - len(dataframe)

        rows_before_missing_cleanup = len(dataframe)
        dataframe = dataframe.dropna()
        missing_values_removed = rows_before_missing_cleanup - len(dataframe)

        self._validate_target_column(dataframe)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(self.output_path, index=False)

        return CleaningSummary(
            input_rows=input_rows,
            output_rows=len(dataframe),
            duplicates_removed=duplicates_removed,
            missing_values_removed=missing_values_removed,
        )

    def _validate_required_columns(self, dataframe: pd.DataFrame) -> None:
        """Ensure required columns exist before processing."""
        missing_columns = self.required_columns - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing_columns)}"
            )

    @staticmethod
    def _validate_target_column(dataframe: pd.DataFrame) -> None:
        """Ensure the fraud label contains only 0 and 1."""
        invalid_values = set(dataframe["Class"].unique()) - {0, 1}

        if invalid_values:
            raise ValueError(
                f"Class column contains invalid values: {sorted(invalid_values)}"
            )