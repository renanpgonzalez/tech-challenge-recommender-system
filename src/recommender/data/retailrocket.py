"""RetailRocket dataset adapters."""

from pathlib import Path

import pandas as pd

from recommender.data.schema import InteractionColumn

RETAILROCKET_EVENTS_COLUMNS = {
    "timestamp": InteractionColumn.TIMESTAMP.value,
    "visitorid": InteractionColumn.USER_ID.value,
    "event": InteractionColumn.EVENT_TYPE.value,
    "itemid": InteractionColumn.ITEM_ID.value,
}

RETAILROCKET_REQUIRED_COLUMNS = set(RETAILROCKET_EVENTS_COLUMNS)


def validate_retailrocket_events_columns(data: pd.DataFrame) -> None:
    """Validate RetailRocket events columns.

    Args:
        data: Raw RetailRocket events dataset.

    Raises:
        ValueError: If required RetailRocket columns are missing.
    """
    missing_columns = sorted(RETAILROCKET_REQUIRED_COLUMNS - set(data.columns))

    if missing_columns:
        message = f"Missing RetailRocket events columns: {', '.join(missing_columns)}"
        raise ValueError(message)


def standardize_retailrocket_events(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize RetailRocket events into the project interaction schema.

    Args:
        data: Raw RetailRocket events dataset.

    Returns:
        Standardized interaction dataset.
    """
    validate_retailrocket_events_columns(data)

    standardized_data = data.rename(columns=RETAILROCKET_EVENTS_COLUMNS)

    return standardized_data[
        [
            InteractionColumn.USER_ID.value,
            InteractionColumn.ITEM_ID.value,
            InteractionColumn.EVENT_TYPE.value,
            InteractionColumn.TIMESTAMP.value,
        ]
    ].copy()


def read_retailrocket_events(path: Path) -> pd.DataFrame:
    """Read RetailRocket events using only required columns.

    Args:
        path: Raw RetailRocket events CSV path.

    Returns:
        Raw RetailRocket events dataframe.
    """
    return pd.read_csv(
        path,
        usecols=list(RETAILROCKET_REQUIRED_COLUMNS),
    )
