"""Tests for interaction feature engineering."""

import pandas as pd
import pytest

from recommender.features.engineering import (
    FeatureColumn,
    add_user_item_indices,
    aggregate_interaction_features,
    build_interaction_features,
    create_index_mapping,
    validate_feature_columns,
)


def make_preprocessed_data() -> pd.DataFrame:
    """Create sample preprocessed interaction data for tests."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 10, 20],
            "event_type": ["view", "addtocart", "transaction"],
            "timestamp": [100, 200, 300],
            "event_weight": [1.0, 3.0, 5.0],
        },
    )


def test_validate_feature_columns_accepts_valid_data() -> None:
    """Validate that feature columns pass validation."""
    validate_feature_columns(make_preprocessed_data())


def test_validate_feature_columns_raises_error_for_missing_columns() -> None:
    """Validate missing feature column handling."""
    data = make_preprocessed_data().drop(columns=["event_weight"])

    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_feature_columns(data)


def test_aggregate_interaction_features() -> None:
    """Validate user-item feature aggregation."""
    result = aggregate_interaction_features(make_preprocessed_data())

    assert len(result) == 2
    assert result.loc[0, "interaction_score"] == 4.0
    assert result.loc[0, "interaction_count"] == 2
    assert result.loc[0, "last_timestamp"] == 200


def test_create_index_mapping_is_deterministic() -> None:
    """Validate deterministic index mapping creation."""
    values = pd.Series([20, 10, 20])

    result = create_index_mapping(values)

    assert result == {"10": 0, "20": 1}


def test_add_user_item_indices() -> None:
    """Validate user and item index creation."""
    aggregated_data = aggregate_interaction_features(make_preprocessed_data())
    result = add_user_item_indices(aggregated_data)

    assert FeatureColumn.USER_INDEX.value in result.columns
    assert FeatureColumn.ITEM_INDEX.value in result.columns
    assert result[FeatureColumn.USER_INDEX.value].tolist() == [0, 1]
    assert result[FeatureColumn.ITEM_INDEX.value].tolist() == [0, 1]


def test_build_interaction_features() -> None:
    """Validate full feature engineering pipeline."""
    result = build_interaction_features(make_preprocessed_data())

    expected_columns = {
        "user_id",
        "item_id",
        "interaction_score",
        "interaction_count",
        "last_timestamp",
        "user_index",
        "item_index",
    }

    assert set(result.columns) == expected_columns
    assert result["interaction_score"].tolist() == [4.0, 5.0]
