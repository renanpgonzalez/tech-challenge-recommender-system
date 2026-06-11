"""Tests for RetailRocket dataset adapters."""

import pandas as pd
import pytest

from recommender.data.retailrocket import (
    standardize_retailrocket_events,
    validate_retailrocket_events_columns,
)


def make_retailrocket_events() -> pd.DataFrame:
    """Create sample RetailRocket events."""
    return pd.DataFrame(
        {
            "timestamp": [1439694000000, 1439695000000],
            "visitorid": [1, 2],
            "event": ["view", "transaction"],
            "itemid": [100, 1000],
            "transactionid": [None, 234],
        },
    )


def test_validate_retailrocket_events_columns_accepts_valid_data() -> None:
    """Validate RetailRocket column validation with valid data."""
    validate_retailrocket_events_columns(make_retailrocket_events())


def test_validate_retailrocket_events_columns_raises_error() -> None:
    """Validate RetailRocket column validation with missing columns."""
    data = make_retailrocket_events().drop(columns=["event"])

    with pytest.raises(ValueError, match="Missing RetailRocket events columns"):
        validate_retailrocket_events_columns(data)


def test_standardize_retailrocket_events() -> None:
    """Validate RetailRocket event standardization."""
    result = standardize_retailrocket_events(make_retailrocket_events())

    assert result.columns.tolist() == [
        "user_id",
        "item_id",
        "event_type",
        "timestamp",
    ]
    assert result["user_id"].tolist() == [1, 2]
    assert result["item_id"].tolist() == [100, 1000]
    assert result["event_type"].tolist() == ["view", "transaction"]
