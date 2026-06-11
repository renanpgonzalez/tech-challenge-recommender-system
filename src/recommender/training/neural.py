"""Neural recommender training utilities."""

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from recommender.models import NeuralRecommender
from recommender.training.dataset import InteractionDataset, get_num_users_items


@dataclass(frozen=True)
class NeuralTrainingConfig:
    """Training configuration for the neural recommender."""

    embedding_dim: int = 32
    hidden_dim: int = 64
    learning_rate: float = 0.001
    epochs: int = 20
    batch_size: int = 64
    validation_fraction: float = 0.2
    patience: int = 3
    min_delta: float = 0.0001
    random_seed: int = 42


@dataclass(frozen=True)
class NeuralTrainingResult:
    """Training result for the neural recommender."""

    train_loss: float
    validation_loss: float
    epochs_trained: int
    best_epoch: int


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible training.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_split_lengths(
    dataset_size: int,
    validation_fraction: float,
) -> tuple[int, int]:
    """Compute train and validation split lengths.

    Args:
        dataset_size: Total dataset size.
        validation_fraction: Fraction used for validation.

    Returns:
        Train and validation lengths.
    """
    if dataset_size < 2:
        return dataset_size, 0

    validation_size = max(1, int(dataset_size * validation_fraction))
    train_size = dataset_size - validation_size

    return train_size, validation_size


def create_data_loaders(
    dataset: InteractionDataset,
    config: NeuralTrainingConfig,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and validation data loaders.

    Args:
        dataset: Interaction dataset.
        config: Training configuration.

    Returns:
        Train and optional validation data loaders.
    """
    train_size, validation_size = compute_split_lengths(
        len(dataset),
        config.validation_fraction,
    )
    generator = torch.Generator().manual_seed(config.random_seed)

    if validation_size == 0:
        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        return train_loader, None

    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)

    return train_loader, validation_loader


def train_one_epoch(
    model: NeuralRecommender,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """Train the model for one epoch.

    Args:
        model: Neural recommender.
        data_loader: Training data loader.
        optimizer: Optimizer.
        loss_fn: Loss function.

    Returns:
        Mean training loss.
    """
    model.train()
    total_loss = 0.0

    for user_indices, item_indices, scores in data_loader:
        optimizer.zero_grad()
        predictions = model(user_indices, item_indices)
        loss = loss_fn(predictions, scores)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(scores)

    return total_loss / len(data_loader.dataset)


def evaluate_loss(
    model: NeuralRecommender,
    data_loader: DataLoader | None,
    loss_fn: nn.Module,
) -> float:
    """Evaluate model loss.

    Args:
        model: Neural recommender.
        data_loader: Evaluation data loader.
        loss_fn: Loss function.

    Returns:
        Mean evaluation loss.
    """
    if data_loader is None:
        return 0.0

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for user_indices, item_indices, scores in data_loader:
            predictions = model(user_indices, item_indices)
            loss = loss_fn(predictions, scores)
            total_loss += float(loss.item()) * len(scores)

    return total_loss / len(data_loader.dataset)


def train_neural_recommender(
    data: pd.DataFrame,
    config: NeuralTrainingConfig,
) -> tuple[NeuralRecommender, NeuralTrainingResult, list[dict[str, float]]]:
    """Train a neural recommender.

    Args:
        data: Feature dataset.
        config: Training configuration.

    Returns:
        Trained model, training result and epoch history.
    """
    set_random_seed(config.random_seed)

    num_users, num_items = get_num_users_items(data)
    model = NeuralRecommender(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    dataset = InteractionDataset(data)
    train_loader, validation_loader = create_data_loaders(dataset, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    return run_training_loop(
        model, train_loader, validation_loader, optimizer, loss_fn, config
    )


def run_training_loop(
    model: NeuralRecommender,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: NeuralTrainingConfig,
) -> tuple[NeuralRecommender, NeuralTrainingResult, list[dict[str, float]]]:
    """Run the neural training loop with early stopping."""
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        validation_loss = evaluate_loss(model, validation_loader, loss_fn) or train_loss
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            },
        )

        print(
            f"Epoch {epoch}/{config.epochs} - "
            f"train_loss={train_loss:.4f} - "
            f"validation_loss={validation_loss:.4f}",
        )

        improved = validation_loss < best_validation_loss - config.min_delta
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    result = NeuralTrainingResult(
        train_loss=history[-1]["train_loss"],
        validation_loss=history[-1]["validation_loss"],
        epochs_trained=len(history),
        best_epoch=best_epoch,
    )

    return model, result, history


def save_neural_model(
    model: NeuralRecommender,
    path: Path,
    config: NeuralTrainingConfig,
    result: NeuralTrainingResult,
    history: list[dict[str, float]],
) -> None:
    """Save neural model artifact.

    Args:
        model: Trained neural recommender.
        path: Output artifact path.
        config: Training configuration.
        result: Training result.
        history: Epoch-level training history.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "num_users": model.user_embedding.num_embeddings,
        "num_items": model.item_embedding.num_embeddings,
        "config": asdict(config),
        "result": asdict(result),
        "history": history,
    }

    torch.save(artifact, path)
