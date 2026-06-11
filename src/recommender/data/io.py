"""Input and output helpers for tabular datasets."""

from pathlib import Path

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    """Create parent directory for a file path.

    Args:
        path: File path whose parent directory must exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def read_dataframe(path: Path) -> pd.DataFrame:
    """Read a dataframe from CSV or Parquet.

    Args:
        path: Input dataset path.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If the input path does not exist.
        ValueError: If the file extension is not supported.
    """
    if not path.exists():
        message = f"Input file not found: {path}"
        raise FileNotFoundError(message)

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    message = f"Unsupported file extension: {suffix}"
    raise ValueError(message)


def write_dataframe(data: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to CSV or Parquet.

    Args:
        data: Dataframe to persist.
        path: Output dataset path.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ensure_parent_dir(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        data.to_csv(path, index=False)
        return

    if suffix == ".parquet":
        data.to_parquet(path, index=False)
        return

    message = f"Unsupported file extension: {suffix}"
    raise ValueError(message)
