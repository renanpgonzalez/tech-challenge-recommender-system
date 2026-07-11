"""Factory for creating recommendation models."""

from typing import Any

from recommender.models.baseline import PopularityRecommender
from recommender.models.neural import NeuralRecommender


class RecommenderFactory:
    """Factory class to instantiate recommendation models."""

    @staticmethod
    def create_recommender(model_type: str, **kwargs: Any) -> Any:
        """Create a recommender instance based on the type.

        Args:
            model_type: Type of the model ('popularity' or 'neural').
            **kwargs: Arguments passed to the model constructor.

        Returns:
            An instance of the requested recommender model.

        Raises:
            ValueError: If the model_type is unknown.
        """
        model_type_lower = model_type.lower()
        if model_type_lower in ("popularity", "popularity_recommender"):
            return PopularityRecommender(**kwargs)
        elif model_type_lower in ("neural", "neural_recommender"):
            return NeuralRecommender(**kwargs)
        else:
            message = f"Unknown recommender model type: {model_type}"
            raise ValueError(message)
