"""Model building, training, evaluation, and persistence."""

import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_and_train_model(X_train, y_train) -> Pipeline:
    """
    Build a character n-gram TF-IDF + Multinomial Naive Bayes pipeline and fit it.

    Uses character n-grams in the (2, 3) range, which suits short text and
    language identification tasks.

    Args:
        X_train: Training text samples (e.g. a pandas Series of strings).
        y_train: Training labels.

    Returns:
        The fitted :class:`sklearn.pipeline.Pipeline` instance.
    """
    pipe = Pipeline(
        [
            ("vectorizer", TfidfVectorizer(analyzer="char", ngram_range=(2, 3))),
            ("multinomialNB", MultinomialNB()),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_model(model, X_test, y_test) -> float:
    """
    Compute accuracy on the test set and print it.

    Args:
        model: A fitted estimator with a ``predict`` method.
        X_test: Test text samples.
        y_test: Ground-truth labels.

    Returns:
        Accuracy score between 0 and 1.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    return accuracy


def save_model(model, filepath: str) -> None:
    """
    Persist the trained pipeline to disk using :mod:`pickle`.

    Creates parent directories if they do not exist.

    Args:
        model: Object to serialize (typically the fitted pipeline).
        filepath: Destination path (e.g. ``.pkl`` file under ``models/``).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
