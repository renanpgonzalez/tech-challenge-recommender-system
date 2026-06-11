"""Tests for interaction feature engineering."""

from pathlib import Path

import pandas as pd
import pytest

from recommender.features.engineering import (
    FeatureColumn,
    FeatureMappings,
    add_user_item_indices,
    aggregate_interaction_features,
    build_interaction_features,
    create_feature_mappings,
    create_index_mapping,
    filter_known_user_items,
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


def test_create_feature_mappings() -> None:
    """Validate feature mapping creation."""
    aggregated_data = aggregate_interaction_features(make_preprocessed_data())

    result = create_feature_mappings(aggregated_data)

    assert result.user_mapping == {"1": 0, "2": 1}
    assert result.item_mapping == {"10": 0, "20": 1}


def test_feature_mappings_save_and_load(tmp_path: Path) -> None:
    """Validate feature mapping persistence."""
    mapping_path = tmp_path / "feature_mappings.json"
    mappings = FeatureMappings(
        user_mapping={"1": 0},
        item_mapping={"10": 0},
    )

    mappings.save(mapping_path)
    result = FeatureMappings.load(mapping_path)

    assert result == mappings


def test_add_user_item_indices() -> None:
    """Validate user and item index creation."""
    aggregated_data = aggregate_interaction_features(make_preprocessed_data())
    result = add_user_item_indices(aggregated_data)

    assert FeatureColumn.USER_INDEX.value in result.columns
    assert FeatureColumn.ITEM_INDEX.value in result.columns
    assert result[FeatureColumn.USER_INDEX.value].tolist() == [0, 1]
    assert result[FeatureColumn.ITEM_INDEX.value].tolist() == [0, 1]


def test_add_user_item_indices_with_existing_mappings() -> None:
    """Validate index creation with reusable mappings."""
    data = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [20],
            "interaction_score": [5.0],
            "interaction_count": [1],
            "last_timestamp": [300],
        },
    )
    mappings = FeatureMappings(
        user_mapping={"1": 0, "2": 1},
        item_mapping={"10": 0, "20": 1},
    )

    result = add_user_item_indices(data, mappings)

    assert result["user_index"].tolist() == [0]
    assert result["item_index"].tolist() == [1]


def test_add_user_item_indices_raises_error_for_unknown_values() -> None:
    """Validate unknown user or item handling."""
    data = pd.DataFrame(
        {
            "user_id": [999],
            "item_id": [20],
            "interaction_score": [5.0],
            "interaction_count": [1],
            "last_timestamp": [300],
        },
    )
    mappings = FeatureMappings(
        user_mapping={"1": 0},
        item_mapping={"20": 0},
    )

    with pytest.raises(ValueError, match="Unknown user_id or item_id"):
        add_user_item_indices(data, mappings)


def test_filter_known_user_items() -> None:
    """Validate filtering unknown users and items."""
    data = pd.DataFrame(
        {
            "user_id": [1, 999],
            "item_id": [10, 20],
            "interaction_score": [4.0, 5.0],
            "interaction_count": [2, 1],
            "last_timestamp": [200, 300],
        },
    )
    mappings = FeatureMappings(
        user_mapping={"1": 0},
        item_mapping={"10": 0},
    )

    result = filter_known_user_items(data, mappings)

    assert len(result) == 1
    assert result.iloc[0]["user_id"] == 1
    assert result.iloc[0]["item_id"] == 10


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


def test_build_interaction_features_drops_unknown_values() -> None:
    """Validate full feature engineering pipeline with unknown filtering."""
    data = pd.DataFrame(
        {
            "user_id": [1, 999],
            "item_id": [10, 20],
            "event_type": ["view", "transaction"],
            "timestamp": [100, 200],
            "event_weight": [1.0, 5.0],
        },
    )
    mappings = FeatureMappings(
        user_mapping={"1": 0},
        item_mapping={"10": 0},
    )

    result = build_interaction_features(data, mappings=mappings, drop_unknown=True)

    assert len(result) == 1
    assert result.iloc[0]["user_index"] == 0
    assert result.iloc[0]["item_index"] == 0
