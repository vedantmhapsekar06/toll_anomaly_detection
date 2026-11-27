"""Candidate models for the two ML detectors.

Deployable candidates are UNSUPERVISED (fitted without labels, because real toll data has
none until analysts confirm alerts). Two SUPERVISED models are included only as a reference:
they show how much headroom the label-free approach leaves on the table.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.model_wrapper import ScoredModel  # noqa: E402

SEED = 42

# family -> (factory(**params) -> ScoredModel, supervised?, name of the capacity knob)
FAMILIES = {
    "isolation_forest": (lambda n_estimators, max_samples: ScoredModel(
        "isolation_forest",
        IsolationForest(n_estimators=n_estimators, max_samples=max_samples, contamination="auto",
                        random_state=SEED, n_jobs=-1),
        params=dict(n_estimators=n_estimators, max_samples=max_samples), seed=SEED),
        False, "n_estimators"),
    "local_outlier_factor": (lambda n_neighbors: ScoredModel(
        "local_outlier_factor", LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, n_jobs=-1),
        params=dict(n_neighbors=n_neighbors), max_fit_rows=5000, seed=SEED),
        False, "n_neighbors"),
    "gaussian_mixture": (lambda n_components: ScoredModel(
        "gaussian_mixture", GaussianMixture(n_components=n_components, covariance_type="full",
                                            reg_covar=1e-3, random_state=SEED),
        params=dict(n_components=n_components), seed=SEED),
        False, "n_components"),
    "one_class_svm": (lambda nu: ScoredModel(
        "one_class_svm", OneClassSVM(nu=nu, gamma="scale"), params=dict(nu=nu),
        max_fit_rows=4000, seed=SEED),
        False, "nu"),
    # ---- supervised references (use labels; NOT deployed) ----
    "logistic_regression": (lambda C: ScoredModel(
        "logistic_regression", LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
        supervised=True, params=dict(C=C), seed=SEED),
        True, "C"),
    "gradient_boosting": (lambda max_iter: ScoredModel(
        "gradient_boosting", HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1,
                                                            early_stopping=False, random_state=SEED),
        supervised=True, params=dict(max_iter=max_iter), seed=SEED),
        True, "max_iter"),
}

# The grid: (family, params). ANCHORS are deliberately low-capacity settings kept only to draw
# the capacity curve (they show what underfitting looks like); they are never selected.
ANCHORS = {("isolation_forest", (("max_samples", 256), ("n_estimators", 25)))}
GRID = (
    [("isolation_forest", dict(n_estimators=n, max_samples=m)) for n, m in
     [(25, 256), (100, 64), (100, 256), (300, 64), (300, 256), (300, 1024), (600, 256)]]
    + [("local_outlier_factor", dict(n_neighbors=35))]
    + [("gaussian_mixture", dict(n_components=k)) for k in (2, 4, 16)]
    + [("one_class_svm", dict(nu=0.05))]
    + [("logistic_regression", dict(C=1.0))]
    + [("gradient_boosting", dict(max_iter=m)) for m in (20, 400)]
)


def is_anchor(family: str, params: dict) -> bool:
    return (family, tuple(sorted(params.items()))) in ANCHORS


def build(family: str, params: dict) -> ScoredModel:
    return FAMILIES[family][0](**params)


def is_supervised(family: str) -> bool:
    return FAMILIES[family][1]


def capacity(family: str, params: dict):
    knob = FAMILIES[family][2]
    return knob, params[knob]


def config_id(family: str, params: dict) -> str:
    return family + "(" + ",".join(f"{k}={v}" for k, v in params.items()) + ")"
