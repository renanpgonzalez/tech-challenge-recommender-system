"""Sampling utilities for training datasets."""

import pandas as pd


def sample_training_data(
    data: pd.DataFrame,
    sample_size: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Sample training data for faster experiment iterations.

    Args:
        data: Full training dataset.
        sample_size: Optional number of rows to sample.
        random_seed: Random seed for reproducibility.

    Returns:
        Sampled or original training dataset.
    """
    if sample_size is None:
        return data

    if sample_size <= 0:
        message = "sample_size must be greater than zero"
        raise ValueError(message)

    if sample_size >= len(data):
        return data

    return data.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
