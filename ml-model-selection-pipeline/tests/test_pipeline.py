import numpy as np

from src.data import load_split
from src.evaluate import pick_threshold, summarize
from src.models import candidates


def test_split_is_stratified_and_disjoint():
    X_train, X_test, y_train, y_test = load_split()
    assert len(X_train) + len(X_test) == 569
    assert abs(y_train.mean() - y_test.mean()) < 0.02
    train_rows = {tuple(r) for r in X_train}
    assert not any(tuple(r) in train_rows for r in X_test)


def test_split_is_deterministic():
    assert np.array_equal(load_split()[1], load_split()[1])


def test_every_candidate_fits_and_predicts_probabilities():
    X_train, X_test, y_train, _ = load_split()
    for name, pipe in candidates().items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2), name
        assert np.allclose(proba.sum(axis=1), 1.0), name


def test_summarize_metrics_in_range():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.4, 0.8, 0.7, 0.9, 0.2])
    m = summarize(y, p)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= m[k] <= 1.0


def test_pick_threshold_reaches_target_recall():
    y = np.array([0] * 20 + [1] * 20)
    p = np.concatenate([np.linspace(0.05, 0.6, 20), np.linspace(0.4, 0.99, 20)])
    t = pick_threshold(y, p, target_recall_class0=0.95)
    pred = (p >= t).astype(int)
    assert np.mean(pred[y == 0] == 0) >= 0.95
