"""Run the interaction feature engineering pipeline."""

import argparse
from pathlib import Path

from recommender.data import read_dataframe, write_dataframe
from recommender.features import (
    FeatureMappings,
    add_user_item_indices,
    aggregate_interaction_features,
    create_feature_mappings,
    filter_known_user_items,
    validate_feature_columns,
)


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
    parser.add_argument("--mapping-input-path", type=Path, default=None)
    parser.add_argument("--mapping-output-path", type=Path, default=None)
    parser.add_argument("--drop-unknown", action="store_true")

    return parser.parse_args()


def load_mappings(mapping_input_path: Path | None) -> FeatureMappings | None:
    """Load mappings when an input path is provided.

    Args:
        mapping_input_path: Optional mapping input path.

    Returns:
        Loaded mappings or None.
    """
    if mapping_input_path is None:
        return None

    return FeatureMappings.load(mapping_input_path)


def run(
    input_path: Path,
    output_path: Path,
    mapping_input_path: Path | None = None,
    mapping_output_path: Path | None = None,
    drop_unknown: bool = False,
) -> None:
    """Run interaction feature engineering.

    Args:
        input_path: Preprocessed interactions dataset path.
        output_path: Feature dataset output path.
        mapping_input_path: Optional existing mapping path.
        mapping_output_path: Optional output mapping path.
        drop_unknown: Whether to drop unknown users and items.
    """
    preprocessed_data = read_dataframe(input_path)
    validate_feature_columns(preprocessed_data)

    aggregated_data = aggregate_interaction_features(preprocessed_data)
    mappings = load_mappings(mapping_input_path) or create_feature_mappings(
        aggregated_data,
    )

    if mapping_input_path is not None and drop_unknown:
        aggregated_data = filter_known_user_items(aggregated_data, mappings)

    feature_data = add_user_item_indices(aggregated_data, mappings)
    write_dataframe(feature_data, output_path)

    if mapping_output_path is not None:
        mappings.save(mapping_output_path)

    print(f"Created {len(feature_data)} feature rows at {output_path}")

    if mapping_output_path is not None:
        print(f"Saved feature mappings to {mapping_output_path}")


def main() -> None:
    """Run the feature engineering command line interface."""
    args = parse_args()
    run(
        input_path=args.input_path,
        output_path=args.output_path,
        mapping_input_path=args.mapping_input_path,
        mapping_output_path=args.mapping_output_path,
        drop_unknown=args.drop_unknown,
    )


if __name__ == "__main__":
    main()
