"""Metrics and threshold selection."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def summarize(y_true, proba, threshold=0.5):
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def pick_threshold(y_true, proba, target_recall_class0=0.95):
    """Choose the lowest threshold in a fixed grid that still catches
    `target_recall_class0` of the malignant cases (label 0), using validation
    predictions only. Label 1 = benign; a higher threshold calls fewer cases
    benign, so the model is more cautious about missing a malignant case.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    malignant = y_true == 0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (proba >= t).astype(int)
        if float(np.mean(pred[malignant] == 0)) >= target_recall_class0:
            return float(t)
    return 0.5
