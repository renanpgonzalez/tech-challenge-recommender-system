"""Model comparison utilities."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetrics:
    """Recommendation metrics for one model."""

    model_name: str
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    coverage_at_k: float


def calculate_relative_difference(
    challenger_value: float,
    baseline_value: float,
) -> float:
    """Calculate relative difference between challenger and baseline.

    Args:
        challenger_value: Challenger model metric value.
        baseline_value: Baseline model metric value.

    Returns:
        Relative difference.
    """
    if baseline_value == 0:
        return 0.0

    return (challenger_value - baseline_value) / baseline_value


def select_winner(
    baseline: ModelMetrics,
    challenger: ModelMetrics,
) -> str:
    """Select model winner using hit rate as the primary metric.

    Args:
        baseline: Baseline model metrics.
        challenger: Challenger model metrics.

    Returns:
        Winner model name.
    """
    if challenger.hit_rate_at_k > baseline.hit_rate_at_k:
        return challenger.model_name

    return baseline.model_name


def build_comparison_rows(
    baseline: ModelMetrics,
    challenger: ModelMetrics,
) -> list[dict[str, float | str]]:
    """Build comparison rows for report generation.

    Args:
        baseline: Baseline model metrics.
        challenger: Challenger model metrics.

    Returns:
        Metric comparison rows.
    """
    return [
        build_metric_row("precision_at_k", baseline, challenger),
        build_metric_row("recall_at_k", baseline, challenger),
        build_metric_row("hit_rate_at_k", baseline, challenger),
        build_metric_row("coverage_at_k", baseline, challenger),
    ]


def build_metric_row(
    metric_name: str,
    baseline: ModelMetrics,
    challenger: ModelMetrics,
) -> dict[str, float | str]:
    """Build one metric comparison row.

    Args:
        metric_name: Metric name.
        baseline: Baseline model metrics.
        challenger: Challenger model metrics.

    Returns:
        Metric comparison row.
    """
    baseline_value = float(getattr(baseline, metric_name))
    challenger_value = float(getattr(challenger, metric_name))

    return {
        "metric": metric_name,
        baseline.model_name: baseline_value,
        challenger.model_name: challenger_value,
        "relative_difference": calculate_relative_difference(
            challenger_value=challenger_value,
            baseline_value=baseline_value,
        ),
    }
