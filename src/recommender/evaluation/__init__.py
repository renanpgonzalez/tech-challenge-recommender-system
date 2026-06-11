"""Evaluation package."""

from recommender.evaluation.baseline import (
    build_catalog_items,
    build_items_by_user,
    build_recommendations_by_user,
    evaluate_popularity_recommender,
)
from recommender.evaluation.comparison import (
    ModelMetrics,
    build_comparison_rows,
    calculate_relative_difference,
    select_winner,
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
from recommender.evaluation.neural import (
    build_item_index_mapping,
    build_neural_recommendations_by_user,
    build_user_index_mapping,
    evaluate_neural_recommender,
    recommend_with_neural_reranking,
    score_candidate_items,
    select_evaluation_users,
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
    "build_item_index_mapping",
    "build_neural_recommendations_by_user",
    "build_user_index_mapping",
    "evaluate_neural_recommender",
    "recommend_with_neural_reranking",
    "score_candidate_items",
    "select_evaluation_users",
    "ModelMetrics",
    "build_comparison_rows",
    "calculate_relative_difference",
    "select_winner",
]
