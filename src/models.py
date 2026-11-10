"""Candidate models. Scaling lives inside each Pipeline, so it is fitted
only on training folds and never leaks information from validation data."""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

from .data import SEED


def candidates():
    return {
        "logistic_regression": Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]
        ),
        "svm_rbf": Pipeline(
            [("scale", StandardScaler()), ("clf", CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False))]
        ),
        "random_forest": Pipeline(
            [("clf", RandomForestClassifier(n_estimators=300, random_state=SEED))]
        ),
        "hist_gradient_boosting": Pipeline(
            [("clf", HistGradientBoostingClassifier(random_state=SEED))]
        ),
    }
