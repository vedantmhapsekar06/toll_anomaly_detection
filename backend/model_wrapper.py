"""One interface for every candidate model, so the live detector does not care which family
won model selection. Pickled models reference this class by import path, so it lives in
`backend` (importable both from train scripts and from the API)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ScoredModel:
    """Scaler + estimator. `score()` always returns "higher = more anomalous".

    * unsupervised estimators (Isolation Forest, LOF, GMM, ...) are fitted without labels and
      scored by the negative of sklearn's `score_samples`;
    * supervised reference models (used only to measure headroom) score with P(anomaly).
    """

    def __init__(self, family: str, estimator, supervised: bool = False,
                 params: dict | None = None, max_fit_rows: int | None = None, seed: int = 42):
        self.family = family
        self.estimator = estimator
        self.supervised = supervised
        self.params = params or {}
        self.max_fit_rows = max_fit_rows
        self.seed = seed
        self.features: list[str] = []
        self.scaler: StandardScaler | None = None
        self.gate_threshold: float | None = None

    def fit(self, X: pd.DataFrame, y=None, gate_col: str | None = None) -> "ScoredModel":
        self.features = list(X.columns)
        A = X.to_numpy(dtype=float)
        self.scaler = StandardScaler().fit(A)
        Z = self.scaler.transform(A)
        if self.max_fit_rows and len(Z) > self.max_fit_rows:          # O(n^2) models: subsample
            keep = np.random.default_rng(self.seed).choice(len(Z), self.max_fit_rows, replace=False)
            Z = Z[keep]
            y = None if y is None else np.asarray(y)[keep]
        if self.supervised:
            self.estimator.fit(Z, np.asarray(y).astype(int))
        else:
            self.estimator.fit(Z)
        # Flood gate: only load ABOVE the training average can be a flood
        self.gate_threshold = float(X[gate_col].mean()) if gate_col else None
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(X[self.features].to_numpy(dtype=float))
        if self.supervised:
            return self.estimator.predict_proba(Z)[:, 1]
        return -self.estimator.score_samples(Z)
