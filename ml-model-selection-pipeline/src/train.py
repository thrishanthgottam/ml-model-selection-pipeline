"""Compare models with cross-validation, pick one, tune its threshold on
out-of-fold predictions, then report once on the held-out test set.

Run:  python -m src.train
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from .data import SEED, load_split
from .evaluate import pick_threshold, summarize
from .models import candidates

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_results = {}
    for name, pipe in candidates().items():
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=["roc_auc", "recall", "precision", "f1"],
        )
        cv_results[name] = {
            k.replace("test_", ""): {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in scores.items() if k.startswith("test_")
        }
        print(f"{name:24s} CV ROC-AUC {cv_results[name]['roc_auc']['mean']:.4f} "
              f"+/- {cv_results[name]['roc_auc']['std']:.4f}")

    best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc"]["mean"])
    best = candidates()[best_name]

    # Threshold is chosen from out-of-fold predictions on TRAINING data only.
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = pick_threshold(y_train, oof, target_recall_class0=0.95)

    best.fit(X_train, y_train)
    proba_test = best.predict_proba(X_test)[:, 1]
    final = {
        "best_model": best_name,
        "default_threshold": summarize(y_test, proba_test, 0.5),
        "tuned_threshold": summarize(y_test, proba_test, threshold),
        "cv_results": cv_results,
        "test_size": int(len(y_test)),
        "seed": SEED,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(final, indent=2))
    joblib.dump(best, MODELS / "best_model.joblib")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (proba_test >= threshold).astype(int),
        display_labels=["malignant", "benign"], ax=ax[0], colorbar=False,
    )
    ax[0].set_title(f"Confusion matrix (threshold={threshold:.2f})")
    RocCurveDisplay.from_predictions(y_test, proba_test, ax=ax[1])
    ax[1].set_title(f"ROC curve - {best_name}")
    fig.tight_layout()
    fig.savefig(REPORTS / "results.png", dpi=120)

    print("\nBest model:", best_name)
    print("Test (threshold 0.50):", final["default_threshold"])
    print(f"Test (threshold {threshold:.2f}):", final["tuned_threshold"])


if __name__ == "__main__":
    main()
