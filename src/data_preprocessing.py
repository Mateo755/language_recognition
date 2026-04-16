"""Text loading and cleaning utilities for the language detection pipeline."""

import re

import pandas as pd


def clean_text(text: str) -> str:
    """
    Normalize a single text string for downstream vectorization.

    Strips noisy punctuation and digits, then lowercases the result.
    """
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    return text.lower()


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
