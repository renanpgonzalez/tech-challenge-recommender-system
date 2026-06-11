"""Generate a model comparison report."""

import argparse
import json
from pathlib import Path

from recommender.data.io import ensure_parent_dir
from recommender.evaluation.comparison import (
    ModelMetrics,
    build_comparison_rows,
    select_winner,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a baseline vs neural model comparison report.",
    )
    parser.add_argument("--baseline-metrics-path", type=Path, required=True)
    parser.add_argument("--neural-metrics-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)

    return parser.parse_args()


def load_metrics(path: Path, model_name: str) -> ModelMetrics:
    """Load recommendation metrics from JSON.

    Args:
        path: Metrics JSON path.
        model_name: Model name.

    Returns:
        Model metrics.
    """
    metrics = json.loads(path.read_text(encoding="utf-8"))

    return ModelMetrics(
        model_name=model_name,
        precision_at_k=float(metrics["precision_at_k"]),
        recall_at_k=float(metrics["recall_at_k"]),
        hit_rate_at_k=float(metrics["hit_rate_at_k"]),
        coverage_at_k=float(metrics["coverage_at_k"]),
    )


def format_percentage(value: float) -> str:
    """Format decimal metric as percentage.

    Args:
        value: Decimal value.

    Returns:
        Percentage string.
    """
    return f"{value:.4%}"


def format_relative_difference(value: float) -> str:
    """Format relative difference as percentage.

    Args:
        value: Relative difference.

    Returns:
        Percentage string with sign.
    """
    return f"{value:+.2%}"


def build_markdown_report(
    baseline: ModelMetrics,
    challenger: ModelMetrics,
) -> str:
    """Build markdown comparison report.

    Args:
        baseline: Baseline model metrics.
        challenger: Challenger model metrics.

    Returns:
        Markdown report.
    """
    rows = build_comparison_rows(baseline, challenger)
    winner = select_winner(baseline, challenger)

    table_rows = "\n".join(
        [
            "| "
            f"{row['metric']} | "
            f"{format_percentage(float(row[baseline.model_name]))} | "
            f"{format_percentage(float(row[challenger.model_name]))} | "
            f"{format_relative_difference(float(row['relative_difference']))} |"
            for row in rows
        ],
    )

    return f"""# Model Comparison Report

## Objective

Compare the popularity baseline recommender against the neural recommender
using ranking metrics at K.

## Compared Models

- Baseline: `{baseline.model_name}`
- Challenger: `{challenger.model_name}`

## Metrics

| Metric | {baseline.model_name} | {challenger.model_name} | Relative Difference |
|---|---:|---:|---:|
{table_rows}

## Decision

Selected model: `{winner}`

The popularity baseline performed better on hit rate, precision and recall
in this experiment.

The neural model increased catalog coverage, but it did not outperform the
baseline in ranking relevance.

## Technical Interpretation

The neural model was trained as a first PyTorch MLP/embedding-based
recommender and evaluated through candidate reranking.

The current version uses a sampled training run and a simple regression
objective over interaction scores.

Future improvements should test larger training samples, ranking-oriented
losses, stronger negative sampling and better candidate generation.
"""


def save_report(report: str, output_path: Path) -> None:
    """Save markdown report.

    Args:
        report: Markdown report.
        output_path: Output path.
    """
    ensure_parent_dir(output_path)
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Run model comparison report generation."""
    args = parse_args()
    baseline = load_metrics(args.baseline_metrics_path, "popularity_baseline")
    challenger = load_metrics(args.neural_metrics_path, "neural_reranker")

    report = build_markdown_report(baseline, challenger)
    save_report(report, args.output_path)

    print(f"Saved model comparison report to {args.output_path}")


if __name__ == "__main__":
    main()
