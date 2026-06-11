"""Evaluate a trained neural recommender."""

import argparse
import json
from pathlib import Path

from recommender.data import read_dataframe
from recommender.data.io import ensure_parent_dir
from recommender.evaluation import evaluate_neural_recommender
from recommender.models import PopularityRecommender, load_neural_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a neural recommender with candidate reranking.",
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--baseline-model-path", type=Path, required=True)
    parser.add_argument("--metrics-output-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-size", type=int, default=100)
    parser.add_argument("--max-users", type=int, default=None)

    return parser.parse_args()


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Save metrics as JSON.

    Args:
        metrics: Evaluation metrics.
        output_path: Metrics output path.
    """
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run(
    train_path: Path,
    test_path: Path,
    model_path: Path,
    baseline_model_path: Path,
    metrics_output_path: Path,
    top_k: int,
    candidate_size: int,
    max_users: int | None,
) -> None:
    """Run neural recommendation evaluation.

    Args:
        train_path: Train feature dataset path.
        test_path: Test feature dataset path.
        model_path: Trained neural model path.
        baseline_model_path: Popularity baseline artifact path.
        metrics_output_path: Metrics output path.
        top_k: Number of recommendations.
        candidate_size: Number of popularity candidates to rerank.
        max_users: Optional maximum number of users to evaluate.
    """
    train_data = read_dataframe(train_path)
    test_data = read_dataframe(test_path)
    model, _ = load_neural_model(model_path)
    baseline_model = PopularityRecommender.load(baseline_model_path)
    candidate_item_ids = baseline_model.top_items[:candidate_size]

    metrics = evaluate_neural_recommender(
        model=model,
        train_data=train_data,
        test_data=test_data,
        candidate_item_ids=candidate_item_ids,
        top_k=top_k,
        max_users=max_users,
    )
    metrics["top_k"] = float(top_k)
    metrics["candidate_size"] = float(candidate_size)
    metrics["max_users"] = float(max_users or 0)

    save_metrics(metrics, metrics_output_path)

    print(f"Saved neural evaluation metrics to {metrics_output_path}")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    """Run the neural evaluation command line interface."""
    args = parse_args()
    run(
        train_path=args.train_path,
        test_path=args.test_path,
        model_path=args.model_path,
        baseline_model_path=args.baseline_model_path,
        metrics_output_path=args.metrics_output_path,
        top_k=args.top_k,
        candidate_size=args.candidate_size,
        max_users=args.max_users,
    )


if __name__ == "__main__":
    main()
