"""Train/test splitting for language detection tabular data."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Split ``Text`` (features) and ``Language`` (labels) into train and test sets.

    Args:
        df: Dataframe containing ``Text`` and ``Language`` columns.
        test_size: Fraction of the dataset to reserve for testing.
        random_state: Seed for reproducible shuffling.

    Returns:
        Tuple ``(X_train, X_test, y_train, y_test)`` as returned by
        :func:`sklearn.model_selection.train_test_split`.
    """
    X = df["Text"]
    y = df["Language"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
