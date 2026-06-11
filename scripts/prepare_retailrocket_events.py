"""Prepare RetailRocket events into the project interaction schema."""

import argparse
from pathlib import Path

from recommender.data import (
    read_retailrocket_events,
    standardize_retailrocket_events,
    write_dataframe,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare RetailRocket events into standardized interactions.",
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)

    return parser.parse_args()


def run(input_path: Path, output_path: Path) -> None:
    """Run RetailRocket event preparation.

    Args:
        input_path: Raw RetailRocket events path.
        output_path: Standardized interactions output path.
    """
    raw_events = read_retailrocket_events(input_path)
    standardized_events = standardize_retailrocket_events(raw_events)
    write_dataframe(standardized_events, output_path)

    print(f"Prepared {len(standardized_events)} RetailRocket events at {output_path}")


def main() -> None:
    """Run the RetailRocket preparation command line interface."""
    args = parse_args()
    run(input_path=args.input_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
