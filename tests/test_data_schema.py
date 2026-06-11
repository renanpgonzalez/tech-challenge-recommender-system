"""Tests for data schema definitions."""

import pytest

from recommender.data.schema import (
    REQUIRED_INTERACTION_COLUMNS,
    EventType,
    InteractionColumn,
    get_event_weight,
)


def test_required_interaction_columns() -> None:
    """Validate required interaction columns."""
    expected_columns = {
        "user_id",
        "item_id",
        "event_type",
        "timestamp",
    }

    assert expected_columns == REQUIRED_INTERACTION_COLUMNS


@pytest.mark.parametrize(
    ("event_type", "expected_weight"),
    [
        (EventType.VIEW, 1.0),
        (EventType.ADD_TO_CART, 3.0),
        (EventType.TRANSACTION, 5.0),
    ],
)
def test_get_event_weight(event_type: EventType, expected_weight: float) -> None:
    """Validate event type weight mapping."""
    assert get_event_weight(event_type) == expected_weight


def test_get_event_weight_is_case_insensitive() -> None:
    """Validate event type normalization."""
    assert get_event_weight("VIEW") == 1.0


def test_get_event_weight_raises_error_for_invalid_event() -> None:
    """Validate unsupported event type handling."""
    with pytest.raises(ValueError, match="Unsupported event type"):
        get_event_weight("unknown")


def test_interaction_column_values() -> None:
    """Validate standardized interaction column values."""
    assert InteractionColumn.USER_ID == "user_id"
    assert InteractionColumn.ITEM_ID == "item_id"
    assert InteractionColumn.EVENT_TYPE == "event_type"
    assert InteractionColumn.TIMESTAMP == "timestamp"
    assert InteractionColumn.EVENT_WEIGHT == "event_weight"
