"""Tests for recommendation evaluation metrics."""

import pytest

from recommender.evaluation.metrics import (
    coverage_at_k,
    get_top_k_items,
    hit_rate_at_k,
    mean_metrics_at_k,
    precision_at_k,
    recall_at_k,
    validate_k,
)


def test_validate_k_raises_error_for_invalid_value() -> None:
    """Validate top-k error handling."""
    with pytest.raises(ValueError, match="k must be greater than zero"):
        validate_k(0)


def test_get_top_k_items_removes_duplicates_preserving_order() -> None:
    """Validate unique top-k recommendation selection."""
    result = get_top_k_items(["10", "20", "10", "30"], k=3)

    assert result == ["10", "20", "30"]


def test_precision_at_k() -> None:
    """Validate precision at k."""
    result = precision_at_k(
        recommended_items=["10", "20", "30"],
        relevant_items={"20", "40"},
        k=3,
    )

    assert result == pytest.approx(1 / 3)


def test_recall_at_k() -> None:
    """Validate recall at k."""
    result = recall_at_k(
        recommended_items=["10", "20", "30"],
        relevant_items={"20", "40"},
        k=3,
    )

    assert result == pytest.approx(1 / 2)


def test_hit_rate_at_k_with_hit() -> None:
    """Validate hit rate when recommendation contains a relevant item."""
    result = hit_rate_at_k(
        recommended_items=["10", "20", "30"],
        relevant_items={"20", "40"},
        k=3,
    )

    assert result == 1.0


def test_hit_rate_at_k_without_hit() -> None:
    """Validate hit rate when recommendation has no relevant item."""
    result = hit_rate_at_k(
        recommended_items=["10", "20", "30"],
        relevant_items={"40", "50"},
        k=3,
    )

    assert result == 0.0


def test_coverage_at_k() -> None:
    """Validate catalog coverage at k."""
    recommendations_by_user = {
        "1": ["10", "20"],
        "2": ["20", "30"],
    }
    catalog_items = {"10", "20", "30", "40"}

    result = coverage_at_k(recommendations_by_user, catalog_items, k=2)

    assert result == pytest.approx(3 / 4)


def test_mean_metrics_at_k() -> None:
    """Validate mean recommendation metrics."""
    recommendations_by_user = {
        "1": ["10", "20"],
        "2": ["30", "40"],
    }
    relevant_items_by_user = {
        "1": {"10", "30"},
        "2": {"40"},
    }
    catalog_items = {"10", "20", "30", "40"}

    result = mean_metrics_at_k(
        recommendations_by_user=recommendations_by_user,
        relevant_items_by_user=relevant_items_by_user,
        catalog_items=catalog_items,
        k=2,
    )

    assert result["precision_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(0.75)
    assert result["hit_rate_at_k"] == pytest.approx(1.0)
    assert result["coverage_at_k"] == pytest.approx(1.0)


def test_mean_metrics_at_k_returns_zero_for_empty_users() -> None:
    """Validate metrics for empty evaluation input."""
    result = mean_metrics_at_k(
        recommendations_by_user={},
        relevant_items_by_user={},
        catalog_items={"10", "20"},
        k=2,
    )

    assert result == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "hit_rate_at_k": 0.0,
        "coverage_at_k": 0.0,
    }
