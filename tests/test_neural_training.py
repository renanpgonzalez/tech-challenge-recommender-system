"""Tests for neural recommender training."""

from pathlib import Path

import pandas as pd
import torch

from recommender.models import NeuralRecommender
from recommender.training.neural import (
    NeuralTrainingConfig,
    NeuralTrainingResult,
    compute_split_lengths,
    save_neural_model,
    train_neural_recommender,
)


def make_feature_data() -> pd.DataFrame:
    """Create sample feature data for neural training tests."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 20, 10, 30],
            "interaction_score": [4.0, 3.0, 2.0, 5.0],
            "interaction_count": [2, 1, 1, 1],
            "last_timestamp": [100, 200, 300, 400],
            "user_index": [0, 0, 1, 1],
            "item_index": [0, 1, 0, 2],
        },
    )


def test_compute_split_lengths() -> None:
    """Validate train and validation split lengths."""
    train_size, validation_size = compute_split_lengths(
        dataset_size=10,
        validation_fraction=0.2,
    )

    assert train_size == 8
    assert validation_size == 2


def test_train_neural_recommender() -> None:
    """Validate neural recommender training."""
    config = NeuralTrainingConfig(
        embedding_dim=4,
        hidden_dim=8,
        learning_rate=0.01,
        epochs=2,
        batch_size=2,
        patience=2,
        random_seed=42,
    )

    model, result, history = train_neural_recommender(make_feature_data(), config)

    assert isinstance(model, NeuralRecommender)
    assert isinstance(result, NeuralTrainingResult)
    assert result.epochs_trained >= 1
    assert len(history) == result.epochs_trained


def test_save_neural_model(tmp_path: Path) -> None:
    """Validate neural model artifact saving."""
    config = NeuralTrainingConfig(embedding_dim=4, hidden_dim=8, epochs=1)
    model, result, history = train_neural_recommender(make_feature_data(), config)
    output_path = tmp_path / "neural_model.pt"

    save_neural_model(
        model=model,
        path=output_path,
        config=config,
        result=result,
        history=history,
    )

    artifact = torch.load(output_path, weights_only=False)

    assert output_path.exists()
    assert "model_state_dict" in artifact
    assert "config" in artifact
    assert "result" in artifact
    assert "history" in artifact
