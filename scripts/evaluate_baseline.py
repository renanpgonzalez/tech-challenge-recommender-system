"""Evaluate the popularity baseline recommender."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from recommender.data import read_dataframe
from recommender.data.io import ensure_parent_dir
from recommender.data.schema import InteractionColumn
from recommender.evaluation import mean_metrics_at_k
from recommender.models import PopularityRecommender


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a popularity-based recommendation baseline.",
    )
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--train-path", type=Path, default=None)

    return parser.parse_args()


def build_items_by_user(data: pd.DataFrame) -> dict[str, set[str]]:
    """Build item sets by user.

    Args:
        data: Feature dataset.

    Returns:
        Mapping from user IDs to interacted item IDs.
    """
    user_column = InteractionColumn.USER_ID.value
    item_column = InteractionColumn.ITEM_ID.value

    items_by_user: dict[str, set[str]] = {}

    for user_id, item_id in (
        data[[user_column, item_column]]
        .astype(str)
        .itertuples(
            index=False,
            name=None,
        )
    ):
        items_by_user.setdefault(user_id, set()).add(item_id)

    return items_by_user


def build_catalog_items(datasets: Sequence[pd.DataFrame]) -> set[str]:
    """Build catalog item set from datasets.

    Args:
        datasets: Feature datasets.

    Returns:
        Unique catalog items.
    """
    item_column = InteractionColumn.ITEM_ID.value
    catalog_items: set[str] = set()

    for data in datasets:
        catalog_items.update(data[item_column].astype(str).unique())

    return catalog_items


def build_recommendations_by_user(
    recommender: PopularityRecommender,
    user_ids: Sequence[str],
    known_items_by_user: Mapping[str, set[str]],
    top_k: int,
) -> dict[str, list[str]]:
    """Build baseline recommendations for each user.

    Args:
        recommender: Fitted popularity recommender.
        user_ids: Users to evaluate.
        known_items_by_user: Items to exclude by user.
        top_k: Number of recommendations to generate.

    Returns:
        Recommendations by user.
    """
    return {
        user_id: recommender.recommend(
            top_n=top_k,
            exclude_items=known_items_by_user.get(user_id, set()),
        )
        for user_id in sorted(user_ids)
    }


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Save metrics as JSON.

    Args:
        metrics: Evaluation metrics.
        output_path: Metrics output path.
    """
    ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run(
    test_path: Path,
    model_path: Path,
    output_path: Path,
    top_k: int,
    train_path: Path | None = None,
) -> None:
    """Run baseline evaluation.

    Args:
        test_path: Test feature dataset path.
        model_path: Trained baseline artifact path.
        output_path: Metrics output path.
        top_k: Number of recommendations to evaluate.
        train_path: Optional train feature dataset path for seen-item exclusion.
    """
    test_data = read_dataframe(test_path)
    train_data = read_dataframe(train_path) if train_path is not None else None

    relevant_items_by_user = build_items_by_user(test_data)
    known_items_by_user = (
        build_items_by_user(train_data) if train_data is not None else {}
    )
    catalog_datasets = [test_data]

    if train_data is not None:
        catalog_datasets.append(train_data)
    catalog_items = build_catalog_items(catalog_datasets)

    recommender = PopularityRecommender.load(model_path)
    recommendations_by_user = build_recommendations_by_user(
        recommender=recommender,
        user_ids=list(relevant_items_by_user),
        known_items_by_user=known_items_by_user,
        top_k=top_k,
    )

    metrics = mean_metrics_at_k(
        recommendations_by_user=recommendations_by_user,
        relevant_items_by_user=relevant_items_by_user,
        catalog_items=catalog_items,
        k=top_k,
    )
    metrics["top_k"] = float(top_k)
    metrics["evaluated_users"] = float(len(relevant_items_by_user))

    save_metrics(metrics, output_path)

    print(f"Saved baseline evaluation metrics to {output_path}")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    """Run the baseline evaluation command line interface."""
    args = parse_args()
    run(
        test_path=args.test_path,
        model_path=args.model_path,
        output_path=args.output_path,
        top_k=args.top_k,
        train_path=args.train_path,
    )


if __name__ == "__main__":
    main()
