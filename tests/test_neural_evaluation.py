"""Tests for neural recommender evaluation."""

import pandas as pd
import pytest

from recommender.evaluation.neural import (
    build_item_index_mapping,
    build_user_index_mapping,
    evaluate_neural_recommender,
    recommend_with_neural_reranking,
    select_evaluation_users,
)
from recommender.models import NeuralRecommender


def make_train_features() -> pd.DataFrame:
    """Create sample train features."""
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "interaction_score": [5.0, 4.0],
            "interaction_count": [2, 1],
            "last_timestamp": [100, 200],
            "user_index": [0, 1],
            "item_index": [0, 1],
        },
    )


def make_test_features() -> pd.DataFrame:
    """Create sample test features."""
    return pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [20],
            "interaction_score": [3.0],
            "interaction_count": [1],
            "last_timestamp": [300],
            "user_index": [0],
            "item_index": [1],
        },
    )


def test_build_user_index_mapping() -> None:
    """Validate user index mapping."""
    result = build_user_index_mapping(make_train_features())

    assert result == {"1": 0, "2": 1}


def test_build_item_index_mapping() -> None:
    """Validate item index mapping."""
    result = build_item_index_mapping(make_train_features())

    assert result == {"10": 0, "20": 1}


def test_select_evaluation_users() -> None:
    """Validate evaluation user selection."""
    result = select_evaluation_users(["2", "1", "3"], max_users=2)

    assert result == ["1", "2"]


def test_select_evaluation_users_raises_error_for_invalid_max_users() -> None:
    """Validate invalid max users handling."""
    with pytest.raises(ValueError, match="max_users must be greater than zero"):
        select_evaluation_users(["1"], max_users=0)


def test_recommend_with_neural_reranking() -> None:
    """Validate neural reranking recommendation output."""
    model = NeuralRecommender(num_users=2, num_items=2, embedding_dim=4, hidden_dim=8)

    result = recommend_with_neural_reranking(
        model=model,
        user_index=0,
        candidate_item_ids=["10", "20"],
        item_index_mapping={"10": 0, "20": 1},
        known_item_ids={"10"},
        top_k=1,
    )

    assert result == ["20"]


def test_evaluate_neural_recommender() -> None:
    """Validate neural recommender evaluation."""
    model = NeuralRecommender(num_users=2, num_items=2, embedding_dim=4, hidden_dim=8)

    result = evaluate_neural_recommender(
        model=model,
        train_data=make_train_features(),
        test_data=make_test_features(),
        candidate_item_ids=["10", "20"],
        top_k=1,
    )

    assert result["hit_rate_at_k"] == pytest.approx(1.0)
