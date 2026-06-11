"""Evaluation package."""

from recommender.evaluation.baseline import (
    build_catalog_items,
    build_items_by_user,
    build_recommendations_by_user,
    evaluate_popularity_recommender,
)
from recommender.evaluation.metrics import (
    coverage_at_k,
    get_top_k_items,
    hit_rate_at_k,
    mean_metrics_at_k,
    precision_at_k,
    recall_at_k,
    validate_k,
)

__all__ = [
    "build_catalog_items",
    "build_items_by_user",
    "build_recommendations_by_user",
    "coverage_at_k",
    "evaluate_popularity_recommender",
    "get_top_k_items",
    "hit_rate_at_k",
    "mean_metrics_at_k",
    "precision_at_k",
    "recall_at_k",
    "validate_k",
]
