"""Tests for baseline evaluation helpers."""

import pandas as pd
import pytest

from recommender.evaluation.baseline import (
    build_catalog_items,
    build_items_by_user,
    build_recommendations_by_user,
    evaluate_popularity_recommender,
)
from recommender.models import PopularityRecommender


def make_train_features() -> pd.DataFrame:
    """Create sample train feature data."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "interaction_score": [5.0, 4.0, 3.0],
            "interaction_count": [2, 1, 1],
            "last_timestamp": [200, 300, 400],
            "user_index": [0, 1, 2],
            "item_index": [0, 1, 2],
        },
    )


def make_test_features() -> pd.DataFrame:
    """Create sample test feature data."""
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [20, 30],
            "interaction_score": [3.0, 5.0],
            "interaction_count": [1, 1],
            "last_timestamp": [500, 600],
            "user_index": [0, 1],
            "item_index": [1, 2],
        },
    )


def test_build_items_by_user() -> None:
    """Validate item set creation by user."""
    result = build_items_by_user(make_test_features())

    assert result == {"1": {"20"}, "2": {"30"}}


def test_build_catalog_items() -> None:
    """Validate catalog item creation from datasets."""
    result = build_catalog_items([make_train_features(), make_test_features()])

    assert result == {"10", "20", "30"}


def test_build_recommendations_by_user() -> None:
    """Validate recommendations by user with known item exclusion."""
    recommender = PopularityRecommender().fit(make_train_features())

    result = build_recommendations_by_user(
        recommender=recommender,
        user_ids=["1", "2"],
        known_items_by_user={"1": {"10"}, "2": {"20"}},
        top_k=2,
    )

    assert result == {
        "1": ["20", "30"],
        "2": ["10", "30"],
    }


def test_evaluate_popularity_recommender() -> None:
    """Validate popularity recommender evaluation."""
    recommender = PopularityRecommender().fit(make_train_features())

    result = evaluate_popularity_recommender(
        recommender=recommender,
        train_data=make_train_features(),
        test_data=make_test_features(),
        top_k=2,
    )

    assert result["precision_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["hit_rate_at_k"] == pytest.approx(1.0)
    assert result["coverage_at_k"] == pytest.approx(1.0)
