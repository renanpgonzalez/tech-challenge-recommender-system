"""Feature engineering package."""

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

__all__ = [
    "FeatureColumn",
    "FeatureMappings",
    "add_user_item_indices",
    "aggregate_interaction_features",
    "build_interaction_features",
    "create_feature_mappings",
    "create_index_mapping",
    "filter_known_user_items",
    "validate_feature_columns",
]
