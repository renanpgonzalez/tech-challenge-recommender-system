"""MLflow helpers for neural recommender experiments."""

from pathlib import Path

import mlflow

from recommender.training import NeuralTrainingConfig, NeuralTrainingResult


def build_neural_params(
    config: NeuralTrainingConfig,
    training_rows: int,
) -> dict[str, float | int | str]:
    """Build MLflow parameters for neural training.

    Args:
        config: Neural training configuration.
        training_rows: Number of training rows.

    Returns:
        MLflow parameter dictionary.
    """
    return {
        "model_type": "neural_recommender",
        "training_rows": training_rows,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "validation_fraction": config.validation_fraction,
        "patience": config.patience,
        "random_seed": config.random_seed,
    }


def build_neural_metrics(
    result: NeuralTrainingResult,
) -> dict[str, float | int]:
    """Build MLflow metrics for neural training.

    Args:
        result: Neural training result.

    Returns:
        MLflow metric dictionary.
    """
    return {
        "train_loss": result.train_loss,
        "validation_loss": result.validation_loss,
        "epochs_trained": result.epochs_trained,
        "best_epoch": result.best_epoch,
    }


def log_neural_training_run(
    params: dict[str, float | int | str],
    metrics: dict[str, float | int],
    model_path: Path,
    metrics_path: Path,
    history_path: Path,
) -> None:
    """Log neural training outputs to MLflow.

    Args:
        params: Experiment parameters.
        metrics: Experiment metrics.
        model_path: Saved model artifact path.
        metrics_path: Saved metrics artifact path.
        history_path: Saved history artifact path.
    """
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(model_path))
    mlflow.log_artifact(str(metrics_path))
    mlflow.log_artifact(str(history_path))
