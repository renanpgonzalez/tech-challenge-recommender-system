"""Tests for neural experiment tracking helpers."""

from pathlib import Path

import mlflow

from recommender.tracking.neural import (
    build_neural_metrics,
    build_neural_params,
    log_neural_training_run,
)
from recommender.training import NeuralTrainingConfig, NeuralTrainingResult


def test_build_neural_params() -> None:
    """Validate neural training parameter creation."""
    config = NeuralTrainingConfig(
        embedding_dim=8,
        hidden_dim=16,
        learning_rate=0.01,
        epochs=5,
        batch_size=2,
        random_seed=42,
    )

    result = build_neural_params(config, training_rows=100)

    assert result["model_type"] == "neural_recommender"
    assert result["training_rows"] == 100
    assert result["embedding_dim"] == 8
    assert result["hidden_dim"] == 16
    assert result["learning_rate"] == 0.01


def test_build_neural_metrics() -> None:
    """Validate neural training metric creation."""
    training_result = NeuralTrainingResult(
        train_loss=0.5,
        validation_loss=0.7,
        epochs_trained=3,
        best_epoch=2,
    )

    result = build_neural_metrics(training_result)

    assert result == {
        "train_loss": 0.5,
        "validation_loss": 0.7,
        "epochs_trained": 3,
        "best_epoch": 2,
    }


def test_log_neural_training_run(tmp_path: Path) -> None:
    """Validate neural training logging to MLflow."""
    model_path = tmp_path / "model.pt"
    metrics_path = tmp_path / "metrics.json"
    history_path = tmp_path / "history.json"
    tracking_db_path = tmp_path / "mlflow_test.db"

    model_path.write_text("model", encoding="utf-8")
    metrics_path.write_text("{}", encoding="utf-8")
    history_path.write_text("[]", encoding="utf-8")

    mlflow.set_tracking_uri(f"sqlite:///{tracking_db_path.as_posix()}")
    mlflow.set_experiment("test-neural-tracking")

    with mlflow.start_run(run_name="test-run"):
        log_neural_training_run(
            params={"model_type": "test"},
            metrics={"train_loss": 1.0},
            model_path=model_path,
            metrics_path=metrics_path,
            history_path=history_path,
        )

    assert tracking_db_path.exists()
