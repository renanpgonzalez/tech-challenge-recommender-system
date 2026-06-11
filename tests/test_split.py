"""Tests for train-test split utilities."""

import pandas as pd
import pytest

from recommender.training.split import (
    TEMP_COUNT_COLUMN,
    TEMP_ORDER_COLUMN,
    add_user_interaction_order,
    chronological_user_split,
    validate_split_columns,
)


def make_interactions() -> pd.DataFrame:
    """Create sample interactions for split tests."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [10, 20, 30, 40, 50],
            "event_type": ["view", "addtocart", "view", "transaction", "view"],
            "timestamp": [100, 200, 100, 300, 100],
            "event_weight": [1.0, 3.0, 1.0, 5.0, 1.0],
        },
    )


def test_validate_split_columns_accepts_valid_data() -> None:
    """Validate that required split columns pass validation."""
    validate_split_columns(make_interactions())


def test_validate_split_columns_raises_error_for_missing_column() -> None:
    """Validate missing split column handling."""
    data = make_interactions().drop(columns=["timestamp"])

    with pytest.raises(ValueError, match="Missing required split columns"):
        validate_split_columns(data)


def test_add_user_interaction_order() -> None:
    """Validate chronological user interaction ordering."""
    result = add_user_interaction_order(make_interactions())

    assert TEMP_ORDER_COLUMN in result.columns
    assert TEMP_COUNT_COLUMN in result.columns
    assert result.loc[0, TEMP_ORDER_COLUMN] == 0
    assert result.loc[1, TEMP_ORDER_COLUMN] == 1


def test_chronological_user_split() -> None:
    """Validate latest interaction per user is used as test."""
    train_data, test_data = chronological_user_split(make_interactions())

    assert len(train_data) == 3
    assert len(test_data) == 2
    assert test_data["item_id"].tolist() == [20, 40]


def test_chronological_user_split_keeps_single_interaction_users_in_train() -> None:
    """Validate users with one interaction remain in train."""
    train_data, test_data = chronological_user_split(make_interactions())

    assert 3 in train_data["user_id"].tolist()
    assert 3 not in test_data["user_id"].tolist()


def test_chronological_user_split_raises_error_for_invalid_test_size() -> None:
    """Validate invalid split parameter handling."""
    with pytest.raises(
        ValueError,
        match="test_interactions_per_user must be greater than zero",
    ):
        chronological_user_split(make_interactions(), test_interactions_per_user=0)
