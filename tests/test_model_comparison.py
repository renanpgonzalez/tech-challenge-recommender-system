"""Tests for model comparison utilities."""

import pytest

from recommender.evaluation.comparison import (
    ModelMetrics,
    build_comparison_rows,
    calculate_relative_difference,
    select_winner,
)


def test_calculate_relative_difference() -> None:
    """Validate relative difference calculation."""
    result = calculate_relative_difference(
        challenger_value=0.15,
        baseline_value=0.10,
    )

    assert result == pytest.approx(0.5)


def test_calculate_relative_difference_with_zero_baseline() -> None:
    """Validate zero baseline handling."""
    result = calculate_relative_difference(
        challenger_value=0.15,
        baseline_value=0.0,
    )

    assert result == 0.0


def test_select_winner_returns_challenger_when_hit_rate_is_better() -> None:
    """Validate challenger selection."""
    baseline = ModelMetrics("baseline", 0.1, 0.1, 0.1, 0.1)
    challenger = ModelMetrics("neural", 0.1, 0.1, 0.2, 0.1)

    result = select_winner(baseline, challenger)

    assert result == "neural"


def test_select_winner_returns_baseline_when_hit_rate_is_not_better() -> None:
    """Validate baseline selection."""
    baseline = ModelMetrics("baseline", 0.1, 0.1, 0.2, 0.1)
    challenger = ModelMetrics("neural", 0.1, 0.1, 0.1, 0.1)

    result = select_winner(baseline, challenger)

    assert result == "baseline"


def test_build_comparison_rows() -> None:
    """Validate comparison row generation."""
    baseline = ModelMetrics("baseline", 0.1, 0.2, 0.3, 0.4)
    challenger = ModelMetrics("neural", 0.2, 0.1, 0.6, 0.8)

    result = build_comparison_rows(baseline, challenger)

    assert len(result) == 4
    assert result[0]["metric"] == "precision_at_k"
    assert result[0]["baseline"] == 0.1
    assert result[0]["neural"] == 0.2
