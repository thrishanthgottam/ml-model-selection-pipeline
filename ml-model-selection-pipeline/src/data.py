"""Data loading and a leakage-safe split."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SEED = 42


def load_split(test_size=0.2, seed=SEED):
    """Return X_train, X_test, y_train, y_test with a stratified split.

    The test set is only used once, at the very end, for the final report.
    """
    data = load_breast_cancer()
    return train_test_split(
        data.data, data.target, test_size=test_size, stratify=data.target, random_state=seed
    )
