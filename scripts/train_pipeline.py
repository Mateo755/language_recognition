"""
End-to-end training pipeline: load data, split, train, evaluate, and save the model.
"""

import sys
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_preprocessing import load_and_clean_data  # noqa: E402
from src.data_split import split_dataset  # noqa: E402
from src.model_training import build_and_train_model, evaluate_model, save_model  # noqa: E402

# Default paths relative to the project root
_DEFAULT_DATA = _ROOT / "data" / "language_detection.csv"
_DEFAULT_MODEL = _ROOT / "models" / "trained_pipeline-0.1.0.pkl"


def run_pipeline(
    data_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
) -> None:
    """
    Execute the full training workflow.

    Args:
        data_path: CSV with ``Text`` and ``Language`` columns. Defaults to
            ``<project>/data/language_detection.csv``.
        model_path: Where to write the pickled pipeline. Defaults to
            ``<project>/models/trained_pipeline-0.1.0.pkl``.
    """
    data_path = data_path or _DEFAULT_DATA
    model_path = model_path or _DEFAULT_MODEL

    print("Pipeline started...")

    # Step 1: load and clean
    print("-> Step 1: Loading and cleaning data")
    df = load_and_clean_data(str(data_path))

    # Step 2: train/test split
    print("-> Step 2: Splitting into train and test sets")
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Step 3: train
    print("-> Step 3: Training model")
    model = build_and_train_model(X_train, y_train)

    # Step 4: evaluate
    print("-> Step 4: Evaluating on the test set")
    evaluate_model(model, X_test, y_test)

    # Step 5: persist
    print("-> Step 5: Saving trained model")
    save_model(model, str(model_path))

    print("Pipeline finished successfully.")


def main() -> None:
    """CLI entry point: run the default pipeline with project default paths."""
    run_pipeline()


if __name__ == "__main__":
    # From project root: python scripts/train_pipeline.py
    # Or: python -m scripts.train_pipeline
    main()
