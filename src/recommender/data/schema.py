"""Data schema definitions for recommendation interactions."""

from enum import StrEnum


class InteractionColumn(StrEnum):
    """Standard column names for user-item interactions."""

    USER_ID = "user_id"
    ITEM_ID = "item_id"
    EVENT_TYPE = "event_type"
    TIMESTAMP = "timestamp"
    EVENT_WEIGHT = "event_weight"


class EventType(StrEnum):
    """Supported e-commerce interaction event types."""

    VIEW = "view"
    ADD_TO_CART = "addtocart"
    TRANSACTION = "transaction"


EVENT_TYPE_WEIGHTS: dict[EventType, float] = {
    EventType.VIEW: 1.0,
    EventType.ADD_TO_CART: 3.0,
    EventType.TRANSACTION: 5.0,
}


REQUIRED_INTERACTION_COLUMNS: set[str] = {
    InteractionColumn.USER_ID,
    InteractionColumn.ITEM_ID,
    InteractionColumn.EVENT_TYPE,
    InteractionColumn.TIMESTAMP,
}


def get_event_weight(event_type: str) -> float:
    """Return the recommendation weight for an event type.

    Args:
        event_type: Raw event type value.

    Returns:
        Event weight used as implicit feedback signal.

    Raises:
        ValueError: If the event type is not supported.
    """
    try:
        normalized_event_type = EventType(event_type.lower())
    except ValueError as error:
        message = f"Unsupported event type: {event_type}"
        raise ValueError(message) from error

    return EVENT_TYPE_WEIGHTS[normalized_event_type]
