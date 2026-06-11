"""Baseline evaluation helpers."""

from collections.abc import Mapping, Sequence

import pandas as pd

from recommender.data.schema import InteractionColumn
from recommender.evaluation.metrics import mean_metrics_at_k
from recommender.models import PopularityRecommender


def build_items_by_user(data: pd.DataFrame) -> dict[str, set[str]]:
    """Build item sets by user.

    Args:
        data: Feature dataset.

    Returns:
        Mapping from user IDs to interacted item IDs.
    """
    user_column = InteractionColumn.USER_ID.value
    item_column = InteractionColumn.ITEM_ID.value
    items_by_user: dict[str, set[str]] = {}

    for user_id, item_id in (
        data[[user_column, item_column]]
        .astype(str)
        .itertuples(
            index=False,
            name=None,
        )
    ):
        items_by_user.setdefault(user_id, set()).add(item_id)

    return items_by_user


def build_catalog_items(datasets: Sequence[pd.DataFrame]) -> set[str]:
    """Build catalog item set from feature datasets.

    Args:
        datasets: Feature datasets.

    Returns:
        Unique catalog items.
    """
    item_column = InteractionColumn.ITEM_ID.value
    catalog_items: set[str] = set()

    for data in datasets:
        catalog_items.update(data[item_column].astype(str).unique())

    return catalog_items


def build_recommendations_by_user(
    recommender: PopularityRecommender,
    user_ids: Sequence[str],
    known_items_by_user: Mapping[str, set[str]],
    top_k: int,
) -> dict[str, list[str]]:
    """Build recommendations for each user.

    Args:
        recommender: Fitted popularity recommender.
        user_ids: Users to evaluate.
        known_items_by_user: Items to exclude by user.
        top_k: Number of recommendations to generate.

    Returns:
        Recommendations by user.
    """
    return {
        user_id: recommender.recommend(
            top_n=top_k,
            exclude_items=known_items_by_user.get(user_id, set()),
        )
        for user_id in sorted(user_ids)
    }


def evaluate_popularity_recommender(
    recommender: PopularityRecommender,
    test_data: pd.DataFrame,
    train_data: pd.DataFrame | None = None,
    top_k: int = 10,
) -> dict[str, float]:
    """Evaluate a popularity recommender.

    Args:
        recommender: Fitted popularity recommender.
        test_data: Test feature dataset.
        train_data: Optional train feature dataset for seen-item exclusion.
        top_k: Number of recommendations to evaluate.

    Returns:
        Recommendation metrics.
    """
    relevant_items_by_user = build_items_by_user(test_data)
    known_items_by_user = (
        build_items_by_user(train_data) if train_data is not None else {}
    )

    catalog_datasets = [test_data]

    if train_data is not None:
        catalog_datasets.append(train_data)

    catalog_items = build_catalog_items(catalog_datasets)
    recommendations_by_user = build_recommendations_by_user(
        recommender=recommender,
        user_ids=list(relevant_items_by_user),
        known_items_by_user=known_items_by_user,
        top_k=top_k,
    )

    return mean_metrics_at_k(
        recommendations_by_user=recommendations_by_user,
        relevant_items_by_user=relevant_items_by_user,
        catalog_items=catalog_items,
        k=top_k,
    )
