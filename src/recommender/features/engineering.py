"""Feature engineering for recommendation interactions."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from recommender.data.schema import InteractionColumn


class FeatureColumn(StrEnum):
    """Standard feature column names."""

    INTERACTION_SCORE = "interaction_score"
    INTERACTION_COUNT = "interaction_count"
    LAST_TIMESTAMP = "last_timestamp"
    USER_INDEX = "user_index"
    ITEM_INDEX = "item_index"


@dataclass(frozen=True)
class FeatureMappings:
    """User and item index mappings for recommendation models."""

    user_mapping: dict[str, int]
    item_mapping: dict[str, int]

    def save(self, path: Path) -> None:
        """Save feature mappings as JSON.

        Args:
            path: Output mapping path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureMappings":
        """Load feature mappings from JSON.

        Args:
            path: Input mapping path.

        Returns:
            Loaded feature mappings.
        """
        artifact = json.loads(path.read_text(encoding="utf-8"))

        return cls(
            user_mapping=artifact["user_mapping"],
            item_mapping=artifact["item_mapping"],
        )


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
    user_column = InteractionColumn.USER_ID.value
    item_column = InteractionColumn.ITEM_ID.value
    weight_column = InteractionColumn.EVENT_WEIGHT.value
    timestamp_column = InteractionColumn.TIMESTAMP.value

    return (
        data.groupby([user_column, item_column], as_index=False)
        .agg(
            interaction_score=(weight_column, "sum"),
            interaction_count=(weight_column, "size"),
            last_timestamp=(timestamp_column, "max"),
        )
        .sort_values([user_column, item_column])
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


def create_feature_mappings(data: pd.DataFrame) -> FeatureMappings:
    """Create user and item mappings from a feature dataset.

    Args:
        data: Aggregated feature dataset.

    Returns:
        Feature mappings.
    """
    user_column = InteractionColumn.USER_ID.value
    item_column = InteractionColumn.ITEM_ID.value

    return FeatureMappings(
        user_mapping=create_index_mapping(data[user_column]),
        item_mapping=create_index_mapping(data[item_column]),
    )


def filter_known_user_items(
    data: pd.DataFrame,
    mappings: FeatureMappings,
) -> pd.DataFrame:
    """Keep only rows with known users and items.

    Args:
        data: Aggregated feature dataset.
        mappings: Existing user and item mappings.

    Returns:
        Dataset containing only known user-item pairs.
    """
    user_column = InteractionColumn.USER_ID.value
    item_column = InteractionColumn.ITEM_ID.value

    known_users = data[user_column].astype(str).isin(mappings.user_mapping)
    known_items = data[item_column].astype(str).isin(mappings.item_mapping)

    return data.loc[known_users & known_items].copy()


def add_user_item_indices(
    data: pd.DataFrame,
    mappings: FeatureMappings | None = None,
) -> pd.DataFrame:
    """Add numeric user and item indices for model training.

    Args:
        data: Aggregated user-item feature dataset.
        mappings: Optional existing mappings.

    Returns:
        Feature dataset with user and item indices.

    Raises:
        ValueError: If unknown users or items are found.
    """
    indexed_data = data.copy()
    feature_mappings = mappings or create_feature_mappings(indexed_data)

    user_column = InteractionColumn.USER_ID.value
    item_column = InteractionColumn.ITEM_ID.value
    user_index_column = FeatureColumn.USER_INDEX.value
    item_index_column = FeatureColumn.ITEM_INDEX.value

    indexed_data[user_index_column] = (
        indexed_data[user_column].astype(str).map(feature_mappings.user_mapping)
    )
    indexed_data[item_index_column] = (
        indexed_data[item_column].astype(str).map(feature_mappings.item_mapping)
    )

    has_unknown_user = indexed_data[user_index_column].isna().any()
    has_unknown_item = indexed_data[item_index_column].isna().any()

    if has_unknown_user or has_unknown_item:
        message = "Unknown user_id or item_id found in feature mappings"
        raise ValueError(message)

    indexed_data[user_index_column] = indexed_data[user_index_column].astype(int)
    indexed_data[item_index_column] = indexed_data[item_index_column].astype(int)

    return indexed_data


def build_interaction_features(
    data: pd.DataFrame,
    mappings: FeatureMappings | None = None,
    drop_unknown: bool = False,
) -> pd.DataFrame:
    """Build model-ready interaction features.

    Args:
        data: Preprocessed interaction dataset.
        mappings: Optional existing user and item mappings.
        drop_unknown: Whether to drop unknown users and items.

    Returns:
        Model-ready interaction feature dataset.
    """
    validate_feature_columns(data)
    aggregated_data = aggregate_interaction_features(data)

    if mappings is not None and drop_unknown:
        aggregated_data = filter_known_user_items(aggregated_data, mappings)

    return add_user_item_indices(aggregated_data, mappings)
