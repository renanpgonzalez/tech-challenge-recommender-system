"""Neural recommender evaluation helpers."""

from collections.abc import Sequence

import pandas as pd
import torch

from recommender.data.schema import InteractionColumn
from recommender.evaluation.baseline import (
    build_catalog_items,
    build_items_by_user,
)
from recommender.evaluation.metrics import mean_metrics_at_k
from recommender.features.engineering import FeatureColumn
from recommender.models import NeuralRecommender


def build_user_index_mapping(data: pd.DataFrame) -> dict[str, int]:
    """Build user ID to user index mapping.

    Args:
        data: Feature dataset.

    Returns:
        User index mapping.
    """
    user_column = InteractionColumn.USER_ID.value
    user_index_column = FeatureColumn.USER_INDEX.value

    return dict(
        zip(
            data[user_column].astype(str),
            data[user_index_column].astype(int),
            strict=False,
        ),
    )


def build_item_index_mapping(data: pd.DataFrame) -> dict[str, int]:
    """Build item ID to item index mapping.

    Args:
        data: Feature dataset.

    Returns:
        Item index mapping.
    """
    item_column = InteractionColumn.ITEM_ID.value
    item_index_column = FeatureColumn.ITEM_INDEX.value

    return dict(
        zip(
            data[item_column].astype(str),
            data[item_index_column].astype(int),
            strict=False,
        ),
    )


def select_evaluation_users(
    user_ids: Sequence[str],
    max_users: int | None = None,
) -> list[str]:
    """Select users for evaluation.

    Args:
        user_ids: Candidate user IDs.
        max_users: Optional maximum number of users.

    Returns:
        Selected user IDs.
    """
    selected_user_ids = sorted(user_ids)

    if max_users is None:
        return selected_user_ids

    if max_users <= 0:
        message = "max_users must be greater than zero"
        raise ValueError(message)

    return selected_user_ids[:max_users]


def score_candidate_items(
    model: NeuralRecommender,
    user_index: int,
    candidate_item_ids: Sequence[str],
    item_index_mapping: dict[str, int],
) -> list[tuple[str, float]]:
    """Score candidate items for one user.

    Args:
        model: Trained neural recommender.
        user_index: User index.
        candidate_item_ids: Candidate item IDs.
        item_index_mapping: Item ID to item index mapping.

    Returns:
        Candidate item IDs and predicted scores.
    """
    valid_item_ids = [
        item_id for item_id in candidate_item_ids if item_id in item_index_mapping
    ]

    if not valid_item_ids:
        return []

    user_indices = torch.full(
        (len(valid_item_ids),),
        fill_value=user_index,
        dtype=torch.long,
    )
    item_indices = torch.tensor(
        [item_index_mapping[item_id] for item_id in valid_item_ids],
        dtype=torch.long,
    )
    scores = model.predict(user_indices, item_indices).tolist()

    return list(zip(valid_item_ids, scores, strict=True))


def recommend_with_neural_reranking(
    model: NeuralRecommender,
    user_index: int,
    candidate_item_ids: Sequence[str],
    item_index_mapping: dict[str, int],
    known_item_ids: set[str],
    top_k: int,
) -> list[str]:
    """Recommend items using neural reranking over candidate items.

    Args:
        model: Trained neural recommender.
        user_index: User index.
        candidate_item_ids: Candidate item IDs.
        item_index_mapping: Item ID to item index mapping.
        known_item_ids: Items already seen by the user.
        top_k: Number of recommendations.

    Returns:
        Ranked recommendations.
    """
    filtered_candidate_ids = [
        item_id for item_id in candidate_item_ids if item_id not in known_item_ids
    ]
    scored_items = score_candidate_items(
        model=model,
        user_index=user_index,
        candidate_item_ids=filtered_candidate_ids,
        item_index_mapping=item_index_mapping,
    )
    ranked_items = sorted(scored_items, key=lambda item: item[1], reverse=True)

    return [item_id for item_id, _ in ranked_items[:top_k]]


def build_neural_recommendations_by_user(
    model: NeuralRecommender,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    candidate_item_ids: Sequence[str],
    top_k: int,
    max_users: int | None = None,
) -> dict[str, list[str]]:
    """Build neural recommendations for evaluation users.

    Args:
        model: Trained neural recommender.
        train_data: Train feature dataset.
        test_data: Test feature dataset.
        candidate_item_ids: Candidate item IDs.
        top_k: Number of recommendations.
        max_users: Optional maximum number of users to evaluate.

    Returns:
        Recommendations by user.
    """
    user_index_mapping = build_user_index_mapping(test_data)
    item_index_mapping = build_item_index_mapping(train_data)
    known_items_by_user = build_items_by_user(train_data)
    relevant_items_by_user = build_items_by_user(test_data)
    evaluation_users = select_evaluation_users(
        list(relevant_items_by_user),
        max_users=max_users,
    )

    return {
        user_id: recommend_with_neural_reranking(
            model=model,
            user_index=user_index_mapping[user_id],
            candidate_item_ids=candidate_item_ids,
            item_index_mapping=item_index_mapping,
            known_item_ids=known_items_by_user.get(user_id, set()),
            top_k=top_k,
        )
        for user_id in evaluation_users
        if user_id in user_index_mapping
    }


def evaluate_neural_recommender(
    model: NeuralRecommender,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    candidate_item_ids: Sequence[str],
    top_k: int = 10,
    max_users: int | None = None,
) -> dict[str, float]:
    """Evaluate neural recommender using candidate reranking.

    Args:
        model: Trained neural recommender.
        train_data: Train feature dataset.
        test_data: Test feature dataset.
        candidate_item_ids: Candidate item IDs.
        top_k: Number of recommendations.
        max_users: Optional maximum number of users to evaluate.

    Returns:
        Recommendation metrics.
    """
    recommendations_by_user = build_neural_recommendations_by_user(
        model=model,
        train_data=train_data,
        test_data=test_data,
        candidate_item_ids=candidate_item_ids,
        top_k=top_k,
        max_users=max_users,
    )
    relevant_items_by_user = build_items_by_user(test_data)
    selected_user_ids = set(recommendations_by_user)

    filtered_relevant_items_by_user = {
        user_id: relevant_items
        for user_id, relevant_items in relevant_items_by_user.items()
        if user_id in selected_user_ids
    }

    return mean_metrics_at_k(
        recommendations_by_user=recommendations_by_user,
        relevant_items_by_user=filtered_relevant_items_by_user,
        catalog_items=build_catalog_items([train_data, test_data]),
        k=top_k,
    )
