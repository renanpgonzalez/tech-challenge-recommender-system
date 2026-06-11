"""Train the PyTorch neural recommender."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from recommender.data import read_dataframe
from recommender.data.io import ensure_parent_dir
from recommender.training import (
    NeuralTrainingConfig,
    save_neural_model,
    train_neural_recommender,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a neural recommender.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--model-output-path", type=Path, required=True)
    parser.add_argument("--metrics-output-path", type=Path, required=True)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)

    return parser.parse_args()


def save_training_metrics(
    metrics: dict[str, float | int],
    output_path: Path,
) -> None:
    """Save training metrics as JSON.

    Args:
        metrics: Training metrics.
        output_path: Metrics output path.
    """
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


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


def run(
    input_path: Path,
    model_output_path: Path,
    metrics_output_path: Path,
    config: NeuralTrainingConfig,
) -> None:
    """Run neural training.

    Args:
        input_path: Feature dataset path.
        model_output_path: Model artifact output path.
        metrics_output_path: Metrics output path.
        config: Neural training configuration.
    """
    feature_data = read_dataframe(input_path)
    model, result, history = train_neural_recommender(feature_data, config)

    save_neural_model(
        model=model,
        path=model_output_path,
        config=config,
        result=result,
        history=history,
    )

    metrics = {
        **asdict(result),
        "training_rows": len(feature_data),
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "learning_rate": config.learning_rate,
    }
    save_training_metrics(metrics, metrics_output_path)

    print(f"Saved neural model to {model_output_path}")
    print(f"Saved neural training metrics to {metrics_output_path}")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    """Run the neural training command line interface."""
    args = parse_args()
    config = build_config(args)

    run(
        input_path=args.input_path,
        model_output_path=args.model_output_path,
        metrics_output_path=args.metrics_output_path,
        config=config,
    )


if __name__ == "__main__":
    main()
