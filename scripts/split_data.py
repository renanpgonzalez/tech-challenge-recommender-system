"""Split preprocessed interactions into train and test datasets."""

import argparse
from pathlib import Path

from recommender.data import read_dataframe, write_dataframe
from recommender.training import chronological_user_split


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Split recommendation interactions into train and test.",
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--train-output-path", type=Path, required=True)
    parser.add_argument("--test-output-path", type=Path, required=True)
    parser.add_argument("--test-interactions-per-user", type=int, default=1)

    return parser.parse_args()


def run(
    input_path: Path,
    train_output_path: Path,
    test_output_path: Path,
    test_interactions_per_user: int,
) -> None:
    """Run train-test split.

    Args:
        input_path: Preprocessed interactions path.
        train_output_path: Train interactions output path.
        test_output_path: Test interactions output path.
        test_interactions_per_user: Latest interactions per user for test.
    """
    interactions = read_dataframe(input_path)
    train_data, test_data = chronological_user_split(
        interactions,
        test_interactions_per_user=test_interactions_per_user,
    )

    write_dataframe(train_data, train_output_path)
    write_dataframe(test_data, test_output_path)

    print(f"Saved {len(train_data)} train interactions to {train_output_path}")
    print(f"Saved {len(test_data)} test interactions to {test_output_path}")


def main() -> None:
    """Run the split command line interface."""
    args = parse_args()
    run(
        input_path=args.input_path,
        train_output_path=args.train_output_path,
        test_output_path=args.test_output_path,
        test_interactions_per_user=args.test_interactions_per_user,
    )


if __name__ == "__main__":
    main()
