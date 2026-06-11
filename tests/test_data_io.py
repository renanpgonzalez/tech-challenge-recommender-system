"""Tests for dataframe input and output helpers."""

import pandas as pd
import pytest

from recommender.data.io import read_dataframe, write_dataframe


def test_write_and_read_csv_dataframe(tmp_path) -> None:
    """Validate dataframe writing and reading with CSV files."""
    output_path = tmp_path / "interactions.csv"
    data = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "event_type": ["view", "transaction"],
            "timestamp": [123456789, 123456790],
        },
    )

    write_dataframe(data, output_path)
    result = read_dataframe(output_path)

    pd.testing.assert_frame_equal(result, data)


def test_write_and_read_parquet_dataframe(tmp_path) -> None:
    """Validate dataframe writing and reading with Parquet files."""
    output_path = tmp_path / "interactions.parquet"
    data = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "event_type": ["view", "transaction"],
            "timestamp": [123456789, 123456790],
        },
    )

    write_dataframe(data, output_path)
    result = read_dataframe(output_path)

    pd.testing.assert_frame_equal(result, data)


def test_read_dataframe_raises_error_for_missing_file(tmp_path) -> None:
    """Validate missing input file handling."""
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Input file not found"):
        read_dataframe(missing_path)


def test_read_dataframe_raises_error_for_unsupported_extension(tmp_path) -> None:
    """Validate unsupported input extension handling."""
    unsupported_path = tmp_path / "interactions.txt"
    unsupported_path.write_text("invalid", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        read_dataframe(unsupported_path)


def test_write_dataframe_raises_error_for_unsupported_extension(tmp_path) -> None:
    """Validate unsupported output extension handling."""
    output_path = tmp_path / "interactions.txt"
    data = pd.DataFrame({"user_id": [1]})

    with pytest.raises(ValueError, match="Unsupported file extension"):
        write_dataframe(data, output_path)
