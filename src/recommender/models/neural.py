"""PyTorch neural recommendation models."""

from pathlib import Path
from typing import Any

import torch
from torch import nn


class NeuralRecommender(nn.Module):
    """Embedding-based neural recommender with an MLP head."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        """Initialize the neural recommender.

        Args:
            num_users: Number of unique users.
            num_items: Number of unique items.
            embedding_dim: Embedding dimension for users and items.
            hidden_dim: Hidden dimension for the MLP layer.
        """
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict interaction scores for user-item pairs.

        Args:
            user_indices: User index tensor.
            item_indices: Item index tensor.

        Returns:
            Predicted interaction scores.
        """
        user_embeddings = self.user_embedding(user_indices)
        item_embeddings = self.item_embedding(item_indices)
        features = torch.cat([user_embeddings, item_embeddings], dim=1)

        return self.mlp(features).squeeze(1)

    def predict(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict scores without tracking gradients.

        Args:
            user_indices: User index tensor.
            item_indices: Item index tensor.

        Returns:
            Predicted interaction scores.
        """
        self.eval()

        with torch.no_grad():
            return self.forward(user_indices, item_indices)


def load_neural_model(path: Path) -> tuple[NeuralRecommender, dict[str, Any]]:
    """Load a saved neural recommender artifact.

    Args:
        path: Saved neural model artifact path.

    Returns:
        Loaded neural model and artifact metadata.
    """
    artifact: dict[str, Any] = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = artifact["model_state_dict"]
    config = artifact["config"]

    num_users = int(
        artifact.get("num_users", state_dict["user_embedding.weight"].shape[0]),
    )
    num_items = int(
        artifact.get("num_items", state_dict["item_embedding.weight"].shape[0]),
    )

    model = NeuralRecommender(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, artifact
