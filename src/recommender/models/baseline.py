"""Baseline recommendation models."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from recommender.data.schema import InteractionColumn
from recommender.features.engineering import FeatureColumn

REQUIRED_BASELINE_COLUMNS: set[str] = {
    InteractionColumn.ITEM_ID.value,
    FeatureColumn.INTERACTION_SCORE.value,
    FeatureColumn.INTERACTION_COUNT.value,
    FeatureColumn.LAST_TIMESTAMP.value,
}


def validate_baseline_columns(data: pd.DataFrame) -> None:
    """Validate required columns for baseline training.

    Args:
        data: Feature dataset.

    Raises:
        ValueError: If required baseline columns are missing.
    """
    missing_columns = sorted(REQUIRED_BASELINE_COLUMNS - set(data.columns))

    if missing_columns:
        message = f"Missing required baseline columns: {', '.join(missing_columns)}"
        raise ValueError(message)


def rank_items_by_popularity(data: pd.DataFrame) -> pd.DataFrame:
    """Rank items by aggregated interaction popularity.

    Args:
        data: Feature dataset.

    Returns:
        Item ranking ordered by popularity signals.
    """
    validate_baseline_columns(data)

    return (
        data.groupby(InteractionColumn.ITEM_ID.value, as_index=False)
        .agg(
            interaction_score=(FeatureColumn.INTERACTION_SCORE.value, "sum"),
            interaction_count=(FeatureColumn.INTERACTION_COUNT.value, "sum"),
            last_timestamp=(FeatureColumn.LAST_TIMESTAMP.value, "max"),
        )
        .sort_values(
            [
                FeatureColumn.INTERACTION_SCORE.value,
                FeatureColumn.INTERACTION_COUNT.value,
                FeatureColumn.LAST_TIMESTAMP.value,
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


from recommender.models.base import BaseRecommender


@dataclass
class PopularityRecommender(BaseRecommender):
    """Popularity-based recommendation baseline."""

    top_items: list[str] = field(default_factory=list)
    item_scores: dict[str, float] = field(default_factory=dict)

    def fit(self, data: pd.DataFrame) -> "PopularityRecommender":
        """Fit the recommender using item popularity.

        Args:
            data: Feature dataset.

        Returns:
            Fitted recommender instance.
        """
        ranking = rank_items_by_popularity(data)
        item_column = InteractionColumn.ITEM_ID.value
        score_column = FeatureColumn.INTERACTION_SCORE.value

        self.top_items = ranking[item_column].astype(str).tolist()
        self.item_scores = dict(
            zip(
                ranking[item_column].astype(str),
                ranking[score_column].astype(float),
                strict=True,
            ),
        )

        return self

    def recommend(
        self,
        top_n: int = 10,
        exclude_items: set[str] | None = None,
    ) -> list[str]:
        """Recommend the most popular items.

        Args:
            top_n: Number of items to recommend.
            exclude_items: Items that should not be recommended.

        Returns:
            Ranked item recommendations.
        """
        excluded_items = exclude_items or set()
        recommendations: list[str] = []

        for item_id in self.top_items:
            if item_id in excluded_items:
                continue

            recommendations.append(item_id)

            if len(recommendations) == top_n:
                break

        return recommendations

    def save(self, path: Path) -> None:
        """Save the recommender artifact as JSON.

        Args:
            path: Output artifact path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model_type": "popularity_recommender",
            "top_items": self.top_items,
            "item_scores": self.item_scores,
        }

        path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PopularityRecommender":
        """Load the recommender artifact from JSON.

        Args:
            path: Input artifact path.

        Returns:
            Loaded recommender.
        """
        artifact: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

        return cls(
            top_items=artifact["top_items"],
            item_scores=artifact["item_scores"],
        )
