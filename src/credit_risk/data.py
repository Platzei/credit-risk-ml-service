from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path("data/raw/cs-training.csv")
TARGET_COLUMN = "SeriousDlqin2yrs"


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the credit risk dataset and validate its target column."""
    data_path = Path(path)

    if not data_path.is_file():
        raise FileNotFoundError(f"Credit data file not found: {data_path}")

    data = pd.read_csv(data_path)
    unnamed_columns = [
        column for column in data.columns if str(column).startswith("Unnamed:")
    ]
    if unnamed_columns:
        data = data.drop(columns=unnamed_columns)

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' is missing from {data_path}"
        )

    return data
