"""Feature engineering for recommendation interactions."""

from enum import StrEnum

import pandas as pd

from recommender.data.schema import InteractionColumn


class FeatureColumn(StrEnum):
    """Standard feature column names."""

    INTERACTION_SCORE = "interaction_score"
    INTERACTION_COUNT = "interaction_count"
    LAST_TIMESTAMP = "last_timestamp"
    USER_INDEX = "user_index"
    ITEM_INDEX = "item_index"


REQUIRED_FEATURE_COLUMNS: set[str] = {
    InteractionColumn.USER_ID.value,
    InteractionColumn.ITEM_ID.value,
    InteractionColumn.EVENT_WEIGHT.value,
    InteractionColumn.TIMESTAMP.value,
}


def validate_feature_columns(data: pd.DataFrame) -> None:
    """Validate required columns for feature engineering.

    Args:
        data: Preprocessed interaction dataset.

    Raises:
        ValueError: If required feature columns are missing.
    """
    missing_columns = sorted(REQUIRED_FEATURE_COLUMNS - set(data.columns))

    if missing_columns:
        message = f"Missing required feature columns: {', '.join(missing_columns)}"
        raise ValueError(message)


def aggregate_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate user-item interactions into recommendation features.

    Args:
        data: Preprocessed interaction dataset.

    Returns:
        Aggregated user-item feature dataset.
    """
    return (
        data.groupby(
            [InteractionColumn.USER_ID.value, InteractionColumn.ITEM_ID.value],
            as_index=False,
        )
        .agg(
            interaction_score=(InteractionColumn.EVENT_WEIGHT.value, "sum"),
            interaction_count=(InteractionColumn.EVENT_WEIGHT.value, "size"),
            last_timestamp=(InteractionColumn.TIMESTAMP.value, "max"),
        )
        .sort_values([InteractionColumn.USER_ID.value, InteractionColumn.ITEM_ID.value])
        .reset_index(drop=True)
    )


def create_index_mapping(values: pd.Series) -> dict[str, int]:
    """Create a deterministic integer index mapping.

    Args:
        values: Entity identifiers.

    Returns:
        Mapping from original identifier to integer index.
    """
    unique_values = values.astype(str).drop_duplicates().sort_values().tolist()

    return {value: index for index, value in enumerate(unique_values)}


def add_user_item_indices(data: pd.DataFrame) -> pd.DataFrame:
    """Add numeric user and item indices for model training.

    Args:
        data: Aggregated user-item feature dataset.

    Returns:
        Feature dataset with user and item indices.
    """
    indexed_data = data.copy()
    user_mapping = create_index_mapping(indexed_data[InteractionColumn.USER_ID.value])
    item_mapping = create_index_mapping(indexed_data[InteractionColumn.ITEM_ID.value])

    indexed_data[FeatureColumn.USER_INDEX.value] = (
        indexed_data[InteractionColumn.USER_ID.value].astype(str).map(user_mapping)
    )
    indexed_data[FeatureColumn.ITEM_INDEX.value] = (
        indexed_data[InteractionColumn.ITEM_ID.value].astype(str).map(item_mapping)
    )

    return indexed_data


def build_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready interaction features.

    Args:
        data: Preprocessed interaction dataset.

    Returns:
        Model-ready interaction feature dataset.
    """
    validate_feature_columns(data)
    aggregated_data = aggregate_interaction_features(data)

    return add_user_item_indices(aggregated_data)
