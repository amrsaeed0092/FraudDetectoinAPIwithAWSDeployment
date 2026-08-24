"""Command-line entry point for cleaning the fraud dataset."""

import logging
from pathlib import Path

from app.data.cleaner import FraudDataCleaner
from app.log.logger import configure_logging


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "creditcard_clean.csv"


def main() -> None:
    """Run the data-cleaning job."""
    configure_logging("INFO")

    cleaner = FraudDataCleaner(
        input_path=RAW_DATA_PATH,
        output_path=CLEAN_DATA_PATH,
    )

    summary = cleaner.clean()

    logging.getLogger(__name__).info(
        "dataset_cleaned",
        extra={
            "input_rows": summary.input_rows,
            "output_rows": summary.output_rows,
            "duplicates_removed": summary.duplicates_removed,
            "missing_values_removed": summary.missing_values_removed,
        },
    )

    print("Cleaning completed successfully.")
    print(f"Input rows: {summary.input_rows}")
    print(f"Output rows: {summary.output_rows}")
    print(f"Duplicates removed: {summary.duplicates_removed}")
    print(f"Missing-value rows removed: {summary.missing_values_removed}")
    print(f"Clean data saved to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()