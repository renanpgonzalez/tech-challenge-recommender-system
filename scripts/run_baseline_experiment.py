"""Run a tracked popularity baseline experiment."""

import argparse
import json
from pathlib import Path

import mlflow

from recommender.data import read_dataframe
from recommender.data.io import ensure_parent_dir
from recommender.evaluation import evaluate_popularity_recommender
from recommender.models import PopularityRecommender
from recommender.tracking import start_mlflow_run


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a tracked popularity baseline experiment.",
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--model-output-path", type=Path, required=True)
    parser.add_argument("--metrics-output-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-name", type=str, default="baseline_popularity")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", type=str, default="product-recommender")

    return parser.parse_args()


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Save metrics as JSON.

    Args:
        metrics: Evaluation metrics.
        output_path: Metrics output path.
    """
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def build_params(
    train_rows: int,
    test_rows: int,
    top_k: int,
) -> dict[str, str | int]:
    """Build MLflow parameters.

    Args:
        train_rows: Number of train rows.
        test_rows: Number of test rows.
        top_k: Number of recommendations evaluated.

    Returns:
        Experiment parameters.
    """
    return {
        "model_type": "popularity_baseline",
        "train_rows": train_rows,
        "test_rows": test_rows,
        "top_k": top_k,
    }


def run(
    train_path: Path,
    test_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    top_k: int,
    run_name: str,
    tracking_uri: str,
    experiment_name: str,
) -> None:
    """Run a tracked baseline experiment.

    Args:
        train_path: Train feature dataset path.
        test_path: Test feature dataset path.
        model_output_path: Model artifact output path.
        metrics_output_path: Metrics output path.
        top_k: Number of recommendations evaluated.
        run_name: MLflow run name.
        tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
    """
    train_data = read_dataframe(train_path)
    test_data = read_dataframe(test_path)

    recommender = PopularityRecommender().fit(train_data)
    recommender.save(model_output_path)

    metrics = evaluate_popularity_recommender(
        recommender=recommender,
        train_data=train_data,
        test_data=test_data,
        top_k=top_k,
    )
    save_metrics(metrics, metrics_output_path)

    params = build_params(
        train_rows=len(train_data),
        test_rows=len(test_data),
        top_k=top_k,
    )

    with start_mlflow_run(
        run_name=run_name,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    ):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_output_path))
        mlflow.log_artifact(str(metrics_output_path))

    print(f"Logged MLflow run: {run_name}")
    print(f"Saved model artifact to {model_output_path}")
    print(f"Saved metrics to {metrics_output_path}")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    """Run the baseline experiment command line interface."""
    args = parse_args()
    run(
        train_path=args.train_path,
        test_path=args.test_path,
        model_output_path=args.model_output_path,
        metrics_output_path=args.metrics_output_path,
        top_k=args.top_k,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
