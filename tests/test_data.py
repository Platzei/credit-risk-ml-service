from pathlib import Path

import pytest

from credit_risk.data import TARGET_COLUMN, load_data


def test_load_data_successfully(tmp_path: Path) -> None:
    csv_path = tmp_path / "credit.csv"
    csv_path.write_text(
        f"{TARGET_COLUMN},age\n0,45\n1,32\n",
        encoding="utf-8",
    )

    data = load_data(csv_path)

    assert list(data.columns) == [TARGET_COLUMN, "age"]
    assert data.shape == (2, 2)


def test_load_data_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Credit data file not found"):
        load_data(missing_path)


def test_load_data_raises_when_target_is_missing(tmp_path: Path) -> None:
    csv_path = tmp_path / "credit.csv"
    csv_path.write_text("age\n45\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Required target column"):
        load_data(csv_path)


def test_load_data_removes_unnamed_index_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "credit.csv"
    csv_path.write_text(
        f"Unnamed: 0,{TARGET_COLUMN},age\n1,0,45\n",
        encoding="utf-8",
    )

    data = load_data(csv_path)

    assert "Unnamed: 0" not in data.columns
    assert list(data.columns) == [TARGET_COLUMN, "age"]
