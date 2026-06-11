"""Feature engineering package."""

from recommender.features.engineering import (
    FeatureColumn,
    add_user_item_indices,
    aggregate_interaction_features,
    build_interaction_features,
    create_index_mapping,
    validate_feature_columns,
)

__all__ = [
    "FeatureColumn",
    "add_user_item_indices",
    "aggregate_interaction_features",
    "build_interaction_features",
    "create_index_mapping",
    "validate_feature_columns",
]
