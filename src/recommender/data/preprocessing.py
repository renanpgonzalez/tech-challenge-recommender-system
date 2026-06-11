"""Preprocessing functions for recommendation interactions."""

import pandas as pd

from recommender.data.schema import (
    REQUIRED_INTERACTION_COLUMNS,
    InteractionColumn,
    get_event_weight,
)


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: set[str] | None = None,
) -> None:
    """Validate whether the dataset contains all required columns.

    Args:
        data: Input interaction dataset.
        required_columns: Expected columns in the dataset.

    Raises:
        ValueError: If required columns are missing.
    """
    columns_to_validate = required_columns or REQUIRED_INTERACTION_COLUMNS
    missing_columns = sorted(columns_to_validate - set(data.columns))

    if missing_columns:
        message = f"Missing required columns: {', '.join(missing_columns)}"
        raise ValueError(message)


def drop_missing_interaction_values(data: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values in required interaction columns.

    Args:
        data: Input interaction dataset.

    Returns:
        Dataset without missing values in required interaction columns.
    """
    return data.dropna(subset=list(REQUIRED_INTERACTION_COLUMNS)).copy()


def normalize_event_types(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize event type values.

    Args:
        data: Input interaction dataset.

    Returns:
        Dataset with normalized event type values.
    """
    normalized_data = data.copy()
    event_type_column = InteractionColumn.EVENT_TYPE

    normalized_data[event_type_column] = (
        normalized_data[event_type_column].astype(str).str.lower()
    )

    return normalized_data


def add_event_weights(data: pd.DataFrame) -> pd.DataFrame:
    """Add event weights based on interaction event types.

    Args:
        data: Input interaction dataset.

    Returns:
        Dataset with event weight column.
    """
    weighted_data = data.copy()
    event_type_column = InteractionColumn.EVENT_TYPE
    event_weight_column = InteractionColumn.EVENT_WEIGHT

    weighted_data[event_weight_column] = weighted_data[event_type_column].apply(
        get_event_weight,
    )

    return weighted_data


def preprocess_interactions(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw user-item interactions.

    Args:
        data: Raw interaction dataset.

    Returns:
        Preprocessed interaction dataset.
    """
    validate_required_columns(data)

    cleaned_data = drop_missing_interaction_values(data)
    normalized_data = normalize_event_types(cleaned_data)

    return add_event_weights(normalized_data)
