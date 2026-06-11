"""PyTorch neural recommendation models."""

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
