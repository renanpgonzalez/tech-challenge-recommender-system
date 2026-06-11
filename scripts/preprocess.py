"""Run the interaction preprocessing pipeline."""

import argparse
from pathlib import Path

from recommender.data import preprocess_interactions, read_dataframe, write_dataframe


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess recommendation interactions.",
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)

    return parser.parse_args()


def run(input_path: Path, output_path: Path) -> None:
    """Run interaction preprocessing.

    Args:
        input_path: Raw interactions dataset path.
        output_path: Preprocessed interactions output path.
    """
    raw_data = read_dataframe(input_path)
    processed_data = preprocess_interactions(raw_data)
    write_dataframe(processed_data, output_path)

    print(f"Preprocessed {len(processed_data)} interactions to {output_path}")


def main() -> None:
    """Run the preprocessing command line interface."""
    args = parse_args()
    run(input_path=args.input_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
