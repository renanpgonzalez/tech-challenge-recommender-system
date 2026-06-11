"""Tests for baseline recommendation models."""

import pandas as pd
import pytest

from recommender.models.baseline import (
    PopularityRecommender,
    rank_items_by_popularity,
    validate_baseline_columns,
)


def make_feature_data() -> pd.DataFrame:
    """Create sample feature data for baseline tests."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 10, 20],
            "interaction_score": [4.0, 1.0, 5.0],
            "interaction_count": [2, 1, 1],
            "last_timestamp": [200, 250, 300],
            "user_index": [0, 1, 2],
            "item_index": [0, 0, 1],
        },
    )


def test_validate_baseline_columns_accepts_valid_data() -> None:
    """Validate that baseline columns pass validation."""
    validate_baseline_columns(make_feature_data())


def test_validate_baseline_columns_raises_error_for_missing_columns() -> None:
    """Validate missing baseline column handling."""
    data = make_feature_data().drop(columns=["interaction_score"])

    with pytest.raises(ValueError, match="Missing required baseline columns"):
        validate_baseline_columns(data)


def test_rank_items_by_popularity() -> None:
    """Validate item popularity ranking."""
    result = rank_items_by_popularity(make_feature_data())

    assert result["item_id"].astype(str).tolist() == ["10", "20"]
    assert result.loc[0, "interaction_score"] == 5.0
    assert result.loc[0, "interaction_count"] == 3


def test_popularity_recommender_fit() -> None:
    """Validate popularity recommender fitting."""
    recommender = PopularityRecommender().fit(make_feature_data())

    assert recommender.top_items == ["10", "20"]
    assert recommender.item_scores == {"10": 5.0, "20": 5.0}


def test_popularity_recommender_recommend_top_n() -> None:
    """Validate top-n recommendations."""
    recommender = PopularityRecommender().fit(make_feature_data())

    assert recommender.recommend(top_n=1) == ["10"]


def test_popularity_recommender_excludes_items() -> None:
    """Validate item exclusion in recommendations."""
    recommender = PopularityRecommender().fit(make_feature_data())

    assert recommender.recommend(top_n=2, exclude_items={"10"}) == ["20"]


def test_popularity_recommender_save_and_load(tmp_path) -> None:
    """Validate saving and loading a popularity recommender."""
    artifact_path = tmp_path / "popularity_model.json"
    recommender = PopularityRecommender().fit(make_feature_data())

    recommender.save(artifact_path)
    loaded_recommender = PopularityRecommender.load(artifact_path)

    assert loaded_recommender.top_items == recommender.top_items
    assert loaded_recommender.item_scores == recommender.item_scores
