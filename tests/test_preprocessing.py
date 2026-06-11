"""Tests for interaction preprocessing."""

import pandas as pd
import pytest

from recommender.data.preprocessing import (
    add_event_weights,
    drop_missing_interaction_values,
    normalize_event_types,
    preprocess_interactions,
    validate_required_columns,
)


def test_validate_required_columns_accepts_valid_dataset() -> None:
    """Validate that datasets with required columns pass validation."""
    data = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "event_type": ["view"],
            "timestamp": [123456789],
        },
    )

    validate_required_columns(data)


def test_validate_required_columns_raises_error_for_missing_columns() -> None:
    """Validate error handling for missing required columns."""
    data = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "timestamp": [123456789],
        },
    )

    with pytest.raises(ValueError, match="Missing required columns: event_type"):
        validate_required_columns(data)


def test_drop_missing_interaction_values() -> None:
    """Validate removal of rows with missing interaction values."""
    data = pd.DataFrame(
        {
            "user_id": [1, None],
            "item_id": [10, 20],
            "event_type": ["view", "addtocart"],
            "timestamp": [123456789, 123456790],
        },
    )

    result = drop_missing_interaction_values(data)

    assert len(result) == 1
    assert result.iloc[0]["user_id"] == 1


def test_normalize_event_types() -> None:
    """Validate event type normalization."""
    data = pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "event_type": ["VIEW", "AddToCart"],
            "timestamp": [123456789, 123456790],
        },
    )

    result = normalize_event_types(data)

    assert result["event_type"].tolist() == ["view", "addtocart"]


def test_add_event_weights() -> None:
    """Validate event weight creation."""
    data = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "event_type": ["view", "addtocart", "transaction"],
            "timestamp": [123456789, 123456790, 123456791],
        },
    )

    result = add_event_weights(data)

    assert result["event_weight"].tolist() == [1.0, 3.0, 5.0]


def test_preprocess_interactions() -> None:
    """Validate full interaction preprocessing pipeline."""
    data = pd.DataFrame(
        {
            "user_id": [1, 2, None],
            "item_id": [10, 20, 30],
            "event_type": ["VIEW", "transaction", "view"],
            "timestamp": [123456789, 123456790, 123456791],
        },
    )

    result = preprocess_interactions(data)

    assert len(result) == 2
    assert result["event_type"].tolist() == ["view", "transaction"]
    assert result["event_weight"].tolist() == [1.0, 5.0]


def test_preprocess_interactions_raises_error_for_invalid_event() -> None:
    """Validate error handling for unsupported event types."""
    data = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [10],
            "event_type": ["unknown"],
            "timestamp": [123456789],
        },
    )

    with pytest.raises(ValueError, match="Unsupported event type"):
        preprocess_interactions(data)
