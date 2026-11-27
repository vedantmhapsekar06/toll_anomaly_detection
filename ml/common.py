"""Shared ML helpers used by select_models.py, train.py and diagnostics.py.

Keeping splits, features, gates, calibration and metrics in ONE place means selection,
training, diagnostics and the live API all measure the same thing the same way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))
from backend import config as cfg  # noqa: E402

# Chronological split. Rows are ordered by time, so no model ever sees the future.
#   train       first 70%. Fits the final model (labels NOT used by the deployable, unsupervised
#               models). Model SELECTION happens inside it with rolling-origin folds (below).
#   calibration next 10%: fits the score -> probability map (labels used)
#   test        last 20%: final report; looked at once, after every choice is locked
SPLIT_FRACS = {"train": 0.70, "calibration": 0.10, "test": 0.20}
N_SELECTION_CHUNKS = 7      # the train window is cut into 7 chunks; folds validate on chunks 3..6
DECISION_P = 0.5
MIN_CAL_POSITIVES = 30


def split_time(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(df)
    i1 = int(n * SPLIT_FRACS["train"])
    i2 = int(n * (SPLIT_FRACS["train"] + SPLIT_FRACS["calibration"]))
    return {"train": df.iloc[:i1], "calibration": df.iloc[i1:i2], "test": df.iloc[i2:]}


def selection_folds(train: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Rolling-origin (expanding window) folds INSIDE the train window: fit on everything
    before a chunk, validate on that chunk. Four folds make model selection far less noisy than
    a single validation block (which holds only ~15 flood bursts)."""
    chunks = np.array_split(np.arange(len(train)), N_SELECTION_CHUNKS)
    return [(train.iloc[:c[0]], train.iloc[c[0]:c[-1] + 1]) for c in chunks[3:]]


def detector_specs(plazas: pd.DataFrame) -> dict[str, dict]:
    """What each ML detector explains, which features it reads and how it is gated."""
    return {
        "api_flood": dict(explains=cfg.DETECTOR_EXPLAINS["api_flood"], featurize=cfg.flood_features,
                          gate_col="ctx_req", events=["plaza_id", "ip_address"], model_path=cfg.FLOOD_MODEL),
        "fare_anomaly": dict(explains=cfg.DETECTOR_EXPLAINS["fare_anomaly"],
                             featurize=lambda d: cfg.fare_features(d, plazas),
                             gate_col=None, events=None, model_path=cfg.FARE_MODEL),
    }


# ---- labels, gate, metrics --------------------------------------------------------------
def labels(df: pd.DataFrame, spec: dict) -> np.ndarray:
    return df["anomaly_type"].isin(spec["explains"]).to_numpy()


def gate_mask(spec: dict, model, X: pd.DataFrame) -> np.ndarray:
    return cfg.above_baseline(model, X) if spec["gate_col"] else np.ones(len(X), dtype=bool)


def prf(y, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[False, True]).ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def ranking_metrics(y, score, gate) -> dict:
    """Threshold-free quality, on the rows the detector is allowed to flag (its gate)."""
    yg, sg = y[gate], score[gate]
    if yg.sum() == 0 or yg.sum() == len(yg):
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}
    return {"pr_auc": round(float(average_precision_score(yg, sg)), 4),
            "roc_auc": round(float(roc_auc_score(yg, sg)), 4)}


def precision_at_prevalence(m: dict, prevalence: float) -> float:
    """Precision at a different anomaly rate, from measured recall and false-positive rate
    (+0.5 smoothing so zero false positives is not read as a rate of exactly zero)."""
    tpr = m["tp"] / max(m["tp"] + m["fn"], 1)
    fpr = (m["fp"] + 0.5) / (m["fp"] + m["tn"] + 1)
    return round(tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence)), 4)


# ---- fit, calibrate, predict ------------------------------------------------------------
def fit_model(spec: dict, builder, train: pd.DataFrame):
    X = spec["featurize"](train)
    model = builder()
    model.fit(X, labels(train, spec) if model.supervised else None, gate_col=spec["gate_col"])
    return model, X


def calibrate(spec: dict, model, cal: pd.DataFrame):
    """Isotonic map from raw score to P(anomaly | score), fitted on gated calibration rows."""
    X = spec["featurize"](cal)
    g, y = gate_mask(spec, model, X), labels(cal, spec)
    if y[g].sum() < MIN_CAL_POSITIVES:
        raise RuntimeError(f"only {int(y[g].sum())} positives in the calibration split")
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(model.score(X)[g], y[g])
    return iso, float(y[g].mean())


def predict(spec: dict, model, iso, df: pd.DataFrame) -> dict:
    X = spec["featurize"](df)
    score = model.score(X)
    gate = gate_mask(spec, model, X)
    p = iso.predict(score)
    return {"X": X, "score": score, "gate": gate, "p": p, "flagged": (p >= DECISION_P) & gate}


def evaluate_split(spec: dict, model, iso, df: pd.DataFrame) -> dict:
    out = predict(spec, model, iso, df)
    y = labels(df, spec)
    return {**prf(y, out["flagged"]), **ranking_metrics(y, out["score"], out["gate"])}


# ---- simple non-ML baselines (every ML detector must beat these) -------------------------
def flood_baseline(cal, test, y_cal, y_te, plazas=None) -> dict:
    """One request-rate threshold, tuned on the labelled calibration split."""
    x_cal, x_te = cal["req_per_min"].to_numpy(float), test["req_per_min"].to_numpy(float)
    cands = np.unique(np.quantile(x_cal, np.linspace(0.5, 0.9995, 400)))
    best = max(cands, key=lambda t: prf(y_cal, x_cal >= t)["f1"])
    return {**prf(y_te, x_te >= best), "rule": f"req_per_min >= {best:.0f}"}


def fare_baseline(cal, test, y_cal, y_te, plazas) -> dict:
    """Naive fixed rule: flag any amount that is not 100% or 50% of the tariff (+-5%)."""
    ratio = cfg.fare_features(test, plazas)["fare_ratio"].to_numpy(float)
    ok = np.isclose(ratio, 1.0, atol=0.05) | np.isclose(ratio, 0.5, atol=0.05)
    return {**prf(y_te, ~ok), "rule": "amount not within 5% of 100% or 50% of tariff"}


def fare_context_baseline(test, y_te, plazas) -> dict:
    """Stronger hand-built rule that sees the SAME context as the model: normal if a standard
    billing level (100%/50%) relative to the tag's own history OR to the plaza's level."""
    f = cfg.fare_features(test, plazas)

    def standard(x):
        return np.isclose(x, 1.0, atol=0.05) | np.isclose(x, 0.5, atol=0.05)
    ok = standard(f["vs_tag"].to_numpy(float)) | standard(f["vs_plaza"].to_numpy(float))
    return {**prf(y_te, ~ok), "rule": "not standard (100%/50%) vs the tag's history nor the plaza's level"}


BASELINES = {"api_flood": flood_baseline, "fare_anomaly": fare_baseline}
