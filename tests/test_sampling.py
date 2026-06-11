"""Tests for training sampling utilities."""

import pandas as pd
import pytest

from recommender.training.sampling import sample_training_data


def make_data() -> pd.DataFrame:
    """Create sample data."""
    return pd.DataFrame({"value": list(range(10))})


def test_sample_training_data_returns_original_when_sample_size_is_none() -> None:
    """Validate no sampling when sample size is not provided."""
    data = make_data()

    result = sample_training_data(data, sample_size=None, random_seed=42)

    assert result.equals(data)


def test_sample_training_data_samples_expected_size() -> None:
    """Validate sampling with expected row count."""
    data = make_data()

    result = sample_training_data(data, sample_size=5, random_seed=42)

    assert len(result) == 5


def test_sample_training_data_returns_original_when_sample_is_larger() -> None:
    """Validate original data is returned when sample size exceeds data size."""
    data = make_data()

    result = sample_training_data(data, sample_size=20, random_seed=42)

    assert result.equals(data)


def test_sample_training_data_raises_error_for_invalid_size() -> None:
    """Validate invalid sample size handling."""
    with pytest.raises(ValueError, match="sample_size must be greater than zero"):
        sample_training_data(make_data(), sample_size=0, random_seed=42)
