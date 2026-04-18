"""Minimal text normalization shared by training and inference (no heavy deps)."""

import re


def clean_text(text: str) -> str:
    """
    Normalize a single text string for downstream vectorization.

    Strips noisy punctuation and digits, then lowercases the result.
    """
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    return text.lower()
