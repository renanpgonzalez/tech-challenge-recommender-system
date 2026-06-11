"""Recommendation evaluation metrics."""

from collections.abc import Mapping, Sequence


def validate_k(k: int) -> None:
    """Validate top-k value.

    Args:
        k: Number of recommendations to evaluate.

    Raises:
        ValueError: If k is not positive.
    """
    if k <= 0:
        message = "k must be greater than zero"
        raise ValueError(message)


def get_top_k_items(items: Sequence[str], k: int) -> list[str]:
    """Return unique top-k items preserving ranking order.

    Args:
        items: Ranked recommended items.
        k: Number of items to keep.

    Returns:
        Unique top-k recommended items.
    """
    validate_k(k)

    return list(dict.fromkeys(items))[:k]


def precision_at_k(
    recommended_items: Sequence[str],
    relevant_items: Sequence[str] | set[str],
    k: int,
) -> float:
    """Compute precision at k.

    Args:
        recommended_items: Ranked recommended items.
        relevant_items: Ground-truth relevant items.
        k: Number of recommendations to evaluate.

    Returns:
        Precision at k score.
    """
    top_k_items = get_top_k_items(recommended_items, k)

    if not top_k_items:
        return 0.0

    relevant_item_set = set(relevant_items)
    hits = len(set(top_k_items) & relevant_item_set)

    return hits / len(top_k_items)


def recall_at_k(
    recommended_items: Sequence[str],
    relevant_items: Sequence[str] | set[str],
    k: int,
) -> float:
    """Compute recall at k.

    Args:
        recommended_items: Ranked recommended items.
        relevant_items: Ground-truth relevant items.
        k: Number of recommendations to evaluate.

    Returns:
        Recall at k score.
    """
    relevant_item_set = set(relevant_items)

    if not relevant_item_set:
        return 0.0

    top_k_items = get_top_k_items(recommended_items, k)
    hits = len(set(top_k_items) & relevant_item_set)

    return hits / len(relevant_item_set)


def hit_rate_at_k(
    recommended_items: Sequence[str],
    relevant_items: Sequence[str] | set[str],
    k: int,
) -> float:
    """Compute hit rate at k.

    Args:
        recommended_items: Ranked recommended items.
        relevant_items: Ground-truth relevant items.
        k: Number of recommendations to evaluate.

    Returns:
        Hit rate at k score.
    """
    top_k_items = get_top_k_items(recommended_items, k)
    relevant_item_set = set(relevant_items)

    return float(bool(set(top_k_items) & relevant_item_set))


def coverage_at_k(
    recommendations_by_user: Mapping[str, Sequence[str]],
    catalog_items: Sequence[str] | set[str],
    k: int,
) -> float:
    """Compute catalog coverage at k.

    Args:
        recommendations_by_user: Ranked recommendations by user.
        catalog_items: Full catalog item set.
        k: Number of recommendations to evaluate per user.

    Returns:
        Catalog coverage at k score.
    """
    catalog_item_set = set(catalog_items)

    if not catalog_item_set:
        return 0.0

    recommended_catalog_items = {
        item
        for recommended_items in recommendations_by_user.values()
        for item in get_top_k_items(recommended_items, k)
    }

    return len(recommended_catalog_items) / len(catalog_item_set)


def mean_metrics_at_k(
    recommendations_by_user: Mapping[str, Sequence[str]],
    relevant_items_by_user: Mapping[str, Sequence[str] | set[str]],
    catalog_items: Sequence[str] | set[str],
    k: int,
) -> dict[str, float]:
    """Compute mean recommendation metrics at k.

    Args:
        recommendations_by_user: Ranked recommendations by user.
        relevant_items_by_user: Ground-truth relevant items by user.
        catalog_items: Full catalog item set.
        k: Number of recommendations to evaluate.

    Returns:
        Mean recommendation metrics.
    """
    validate_k(k)

    user_ids = sorted(relevant_items_by_user)

    if not user_ids:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "coverage_at_k": 0.0,
        }

    precision_scores = [
        precision_at_k(
            recommendations_by_user.get(user_id, []),
            relevant_items_by_user[user_id],
            k,
        )
        for user_id in user_ids
    ]
    recall_scores = [
        recall_at_k(
            recommendations_by_user.get(user_id, []),
            relevant_items_by_user[user_id],
            k,
        )
        for user_id in user_ids
    ]
    hit_rate_scores = [
        hit_rate_at_k(
            recommendations_by_user.get(user_id, []),
            relevant_items_by_user[user_id],
            k,
        )
        for user_id in user_ids
    ]

    return {
        "precision_at_k": sum(precision_scores) / len(precision_scores),
        "recall_at_k": sum(recall_scores) / len(recall_scores),
        "hit_rate_at_k": sum(hit_rate_scores) / len(hit_rate_scores),
        "coverage_at_k": coverage_at_k(recommendations_by_user, catalog_items, k),
    }
