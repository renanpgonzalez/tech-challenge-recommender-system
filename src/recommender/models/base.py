"""Abstract base class for recommender models."""

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class representing a generic recommender model."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseRecommender":
        """Fit the model on interaction data.

        Args:
            data: Input features/interactions dataframe.

        Returns:
            Fitted recommender instance.
        """
        pass

    @abstractmethod
    def recommend(
        self,
        top_n: int = 10,
        exclude_items: set[str] | None = None,
    ) -> list[str]:
        """Generate recommendation outputs.

        Args:
            top_n: Number of items to recommend.
            exclude_items: Optional set of item IDs to filter out.

        Returns:
            List of recommended item IDs.
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the model artifacts to disk.

        Args:
            path: Destination file path.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseRecommender":
        """Load the model artifacts from disk.

        Args:
            path: Source file path.

        Returns:
            Loaded recommender instance.
        """
        pass
