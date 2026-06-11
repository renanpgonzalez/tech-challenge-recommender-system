"""Tests for PyTorch training datasets."""

import pandas as pd
import torch

from recommender.training.dataset import InteractionDataset, get_num_users_items


def make_feature_data() -> pd.DataFrame:
    """Create sample feature data for dataset tests."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": [10, 20, 30],
            "interaction_score": [4.0, 5.0, 2.0],
            "interaction_count": [2, 1, 1],
            "last_timestamp": [100, 200, 300],
            "user_index": [0, 1, 2],
            "item_index": [0, 1, 2],
        },
    )


def test_interaction_dataset_length() -> None:
    """Validate interaction dataset length."""
    dataset = InteractionDataset(make_feature_data())

    assert len(dataset) == 3


def test_interaction_dataset_item() -> None:
    """Validate interaction dataset sample."""
    dataset = InteractionDataset(make_feature_data())
    user_index, item_index, score = dataset[0]

    assert user_index == torch.tensor(0)
    assert item_index == torch.tensor(0)
    assert score == torch.tensor(4.0)


def test_get_num_users_items() -> None:
    """Validate user and item counts."""
    num_users, num_items = get_num_users_items(make_feature_data())

    assert num_users == 3
    assert num_items == 3
