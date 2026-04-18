"""Text loading and cleaning utilities for the language detection pipeline."""

import pandas as pd

from src.text_cleaning import clean_text


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file and apply :func:`clean_text` to the ``Text`` column.

    Args:
        filepath: Path to a CSV with at least a ``Text`` column.

    Returns:
        A copy of the dataframe with cleaned ``Text`` values.
    """
    df = pd.read_csv(filepath)
    df["Text"] = df["Text"].apply(clean_text)
    return df
