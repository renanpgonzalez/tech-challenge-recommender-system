"""PyTorch datasets for recommendation training."""

import pandas as pd
import torch
from torch.utils.data import Dataset

from recommender.features.engineering import FeatureColumn


class InteractionDataset(Dataset):
    """PyTorch dataset for user-item interaction features."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Initialize the interaction dataset.

        Args:
            data: Feature dataset with user, item and score columns.
        """
        self.user_indices = torch.tensor(
            data[FeatureColumn.USER_INDEX.value].to_numpy(),
            dtype=torch.long,
        )
        self.item_indices = torch.tensor(
            data[FeatureColumn.ITEM_INDEX.value].to_numpy(),
            dtype=torch.long,
        )
        self.scores = torch.tensor(
            data[FeatureColumn.INTERACTION_SCORE.value].to_numpy(),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.scores)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one training sample.

        Args:
            index: Sample index.

        Returns:
            User index, item index and interaction score.
        """
        return (
            self.user_indices[index],
            self.item_indices[index],
            self.scores[index],
        )


def get_num_users_items(data: pd.DataFrame) -> tuple[int, int]:
    """Return the number of users and items for embedding layers.

    Args:
        data: Feature dataset.

    Returns:
        Number of users and items.
    """
    num_users = int(data[FeatureColumn.USER_INDEX.value].max()) + 1
    num_items = int(data[FeatureColumn.ITEM_INDEX.value].max()) + 1

    return num_users, num_items
