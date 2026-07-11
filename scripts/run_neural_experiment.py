"""Run a tracked neural recommender experiment."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from recommender.data import read_dataframe
from recommender.data.io import ensure_parent_dir
from recommender.tracking import (
    build_neural_metrics,
    build_neural_params,
    log_neural_training_run,
    start_mlflow_run,
)
from recommender.training import (
    NeuralTrainingConfig,
    sample_training_data,
    save_neural_model,
    train_neural_recommender,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a tracked neural recommender experiment.",
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--model-output-path", type=Path, required=True)
    parser.add_argument("--metrics-output-path", type=Path, required=True)
    parser.add_argument("--history-output-path", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="neural_recommender_v1")
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", type=str, default="product-recommender")
    parser.add_argument("--sample-size", type=int, default=None)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> NeuralTrainingConfig:
    """Build neural training configuration.

    Args:
        args: Parsed command line arguments.

    Returns:
        Neural training configuration.
    """
    return NeuralTrainingConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        random_seed=args.random_seed,
    )


def save_json(data: object, output_path: Path) -> None:
    """Save data as JSON.

    Args:
        data: Serializable data.
        output_path: Output JSON path.
    """
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_neural_artifacts(
    model: Any,
    model_output_path: Path,
    config: NeuralTrainingConfig,
    result: Any,
    history: list[dict[str, float]],
    metrics: dict[str, float],
    metrics_output_path: Path,
    history_output_path: Path,
) -> None:
    """Save all neural training artifacts to disk.

    Args:
        model: Trained neural recommender.
        model_output_path: Model artifact output path.
        config: Neural training configuration.
        result: Training result.
        history: Epoch history list.
        metrics: Evaluation metrics.
        metrics_output_path: Metrics output path.
        history_output_path: History output path.
    """
    save_neural_model(
        model=model,
        path=model_output_path,
        config=config,
        result=result,
        history=history,
    )
    save_json({**metrics, **asdict(config)}, metrics_output_path)
    save_json(history, history_output_path)


def log_neural_experiment_to_mlflow(
    run_name: str,
    tracking_uri: str,
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    model_output_path: Path,
    metrics_output_path: Path,
    history_output_path: Path,
) -> None:
    """Log neural experiment run to MLflow.

    Args:
        run_name: MLflow run name.
        tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        params: Experiment parameters.
        metrics: Evaluation metrics.
        model_output_path: Model artifact path.
        metrics_output_path: Metrics path.
        history_output_path: History path.
    """
    with start_mlflow_run(
        run_name=run_name,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    ):
        log_neural_training_run(
            params=params,
            metrics=metrics,
            model_path=model_output_path,
            metrics_path=metrics_output_path,
            history_path=history_output_path,
        )


def run(
    train_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    history_output_path: Path,
    config: NeuralTrainingConfig,
    run_name: str,
    tracking_uri: str,
    experiment_name: str,
    sample_size: int | None,
) -> None:
    """Run a tracked neural training experiment."""
    train_data = read_dataframe(train_path)
    original_training_rows = len(train_data)
    train_data = sample_training_data(
        data=train_data,
        sample_size=sample_size,
        random_seed=config.random_seed,
    )

    print(f"Original training rows: {original_training_rows}")
    print(f"Effective training rows: {len(train_data)}")
    model, result, history = train_neural_recommender(train_data, config)

    metrics = build_neural_metrics(result)
    save_neural_artifacts(
        model,
        model_output_path,
        config,
        result,
        history,
        metrics,
        metrics_output_path,
        history_output_path,
    )

    params = build_neural_params(
        config=config,
        training_rows=len(train_data),
        original_training_rows=original_training_rows,
        sample_size=sample_size,
    )
    log_neural_experiment_to_mlflow(
        run_name,
        tracking_uri,
        experiment_name,
        params,
        metrics,
        model_output_path,
        metrics_output_path,
        history_output_path,
    )

    print(f"Logged MLflow run: {run_name}")
    print(f"Saved neural model to {model_output_path}")
    print(f"Saved neural metrics to {metrics_output_path}")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    """Run the neural experiment command line interface."""
    args = parse_args()
    config = build_config(args)

    run(
        train_path=args.train_path,
        model_output_path=args.model_output_path,
        metrics_output_path=args.metrics_output_path,
        history_output_path=args.history_output_path,
        config=config,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
