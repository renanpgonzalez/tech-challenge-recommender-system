"""Train-test split utilities for recommendation interactions."""

import pandas as pd

from recommender.data.schema import InteractionColumn

TEMP_ORDER_COLUMN = "_interaction_order"
TEMP_COUNT_COLUMN = "_interaction_count"

REQUIRED_SPLIT_COLUMNS: set[str] = {
    InteractionColumn.USER_ID.value,
    InteractionColumn.TIMESTAMP.value,
}


def validate_split_columns(data: pd.DataFrame) -> None:
    """Validate required columns for train-test split.

    Args:
        data: Preprocessed interaction dataset.

    Raises:
        ValueError: If required columns are missing.
    """
    missing_columns = sorted(REQUIRED_SPLIT_COLUMNS - set(data.columns))

    if missing_columns:
        message = f"Missing required split columns: {', '.join(missing_columns)}"
        raise ValueError(message)


def add_user_interaction_order(data: pd.DataFrame) -> pd.DataFrame:
    """Add chronological interaction order by user.

    Args:
        data: Preprocessed interaction dataset.

    Returns:
        Dataset with temporary order and count columns.
    """
    user_column = InteractionColumn.USER_ID.value
    timestamp_column = InteractionColumn.TIMESTAMP.value
    ordered_data = data.sort_values([user_column, timestamp_column]).copy()

    ordered_data[TEMP_ORDER_COLUMN] = ordered_data.groupby(user_column).cumcount()
    ordered_data[TEMP_COUNT_COLUMN] = ordered_data.groupby(user_column)[
        user_column
    ].transform("size")

    return ordered_data


def chronological_user_split(
    data: pd.DataFrame,
    test_interactions_per_user: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions using the latest interactions per user as test.

    Args:
        data: Preprocessed interaction dataset.
        test_interactions_per_user: Number of latest interactions per user for test.

    Returns:
        Train and test datasets.
    """
    if test_interactions_per_user <= 0:
        message = "test_interactions_per_user must be greater than zero"
        raise ValueError(message)

    validate_split_columns(data)
    ordered_data = add_user_interaction_order(data)

    test_start = ordered_data[TEMP_COUNT_COLUMN] - test_interactions_per_user
    test_mask = (ordered_data[TEMP_COUNT_COLUMN] > test_interactions_per_user) & (
        ordered_data[TEMP_ORDER_COLUMN] >= test_start
    )

    columns_to_drop = [TEMP_ORDER_COLUMN, TEMP_COUNT_COLUMN]
    train_data = ordered_data.loc[~test_mask].drop(columns=columns_to_drop)
    test_data = ordered_data.loc[test_mask].drop(columns=columns_to_drop)

    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)
