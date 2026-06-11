"""Run the interaction feature engineering pipeline."""

import argparse
from pathlib import Path

from recommender.data import read_dataframe, write_dataframe
from recommender.features import build_interaction_features


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build recommendation interaction features.",
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)

    return parser.parse_args()


def run(input_path: Path, output_path: Path) -> None:
    """Run interaction feature engineering.

    Args:
        input_path: Preprocessed interactions dataset path.
        output_path: Feature dataset output path.
    """
    preprocessed_data = read_dataframe(input_path)
    feature_data = build_interaction_features(preprocessed_data)
    write_dataframe(feature_data, output_path)

    print(f"Created {len(feature_data)} feature rows at {output_path}")


def main() -> None:
    """Run the feature engineering command line interface."""
    args = parse_args()
    run(input_path=args.input_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
