"""Train the popularity baseline recommender."""

import argparse
from pathlib import Path

from recommender.data import read_dataframe
from recommender.models import PopularityRecommender


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a popularity-based recommendation baseline.",
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)

    return parser.parse_args()


def run(input_path: Path, output_path: Path, top_n: int) -> None:
    """Run baseline training.

    Args:
        input_path: Feature dataset path.
        output_path: Baseline model artifact path.
        top_n: Number of top recommendations to preview.
    """
    feature_data = read_dataframe(input_path)
    recommender = PopularityRecommender().fit(feature_data)
    recommender.save(output_path)

    recommendations = recommender.recommend(top_n=top_n)
    print(f"Saved popularity baseline to {output_path}")
    print(f"Top {top_n} recommendations: {recommendations}")


def main() -> None:
    """Run the baseline training command line interface."""
    args = parse_args()
    run(
        input_path=args.input_path,
        output_path=args.output_path,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
