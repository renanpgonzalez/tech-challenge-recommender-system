"""Tests for PyTorch neural recommendation models."""

import torch

from recommender.models import NeuralRecommender


def test_neural_recommender_forward_shape() -> None:
    """Validate neural recommender forward output shape."""
    model = NeuralRecommender(
        num_users=3,
        num_items=4,
        embedding_dim=8,
        hidden_dim=16,
    )
    user_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    item_indices = torch.tensor([1, 2, 3], dtype=torch.long)

    predictions = model(user_indices, item_indices)

    assert predictions.shape == torch.Size([3])


def test_neural_recommender_predict_shape() -> None:
    """Validate neural recommender prediction output shape."""
    model = NeuralRecommender(
        num_users=3,
        num_items=4,
        embedding_dim=8,
        hidden_dim=16,
    )
    user_indices = torch.tensor([0, 1], dtype=torch.long)
    item_indices = torch.tensor([1, 2], dtype=torch.long)

    predictions = model.predict(user_indices, item_indices)

    assert predictions.shape == torch.Size([2])
    assert predictions.requires_grad is False
