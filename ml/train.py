"""Train the two Isolation Forest detectors, calibrate their scores, write metrics.json.

Run from the project root:  python ml/train.py

Split (chronological, so the model never sees the future):
    first 60% of time -> fit the Isolation Forest   (labels NOT used)
    next  15%         -> fit the score calibrator   (labels used)
    last  25%         -> test, never touched while fitting

Calibration: isotonic regression maps a raw Isolation Forest score to
P(anomaly | score), measured on the calibration split. A row is flagged when
that probability is >= 0.5.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import anomaly_detector as ad  # noqa: E402
from backend import config as cfg  # noqa: E402

TRAIN_FRAC, CAL_FRAC = 0.60, 0.15
DECISION_P = 0.5


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("iforest", IsolationForest(n_estimators=300, contamination="auto",
                                    random_state=cfg.SEED, n_jobs=-1)),
    ])


def anomaly_score(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous (sklearn's score_samples is higher = more normal)."""
    return -pipe.score_samples(X)


def prf(y, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[False, True]).ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def run_detector(name, featurize, gate, explains, train, cal, test):
    pipe = build_pipeline().fit(featurize(train))

    # Gate: rows the detector is allowed to flag at all (e.g. flood = above-average load).
    # Calibration and evaluation both use only gated rows, matching live behaviour.
    x_cal, x_te = featurize(cal), featurize(test)
    g_cal = gate(pipe, x_cal) if gate else np.ones(len(cal), dtype=bool)
    g_te = gate(pipe, x_te) if gate else np.ones(len(test), dtype=bool)

    s_cal = anomaly_score(pipe, x_cal)
    y_cal = cal["anomaly_type"].isin(explains).to_numpy()
    assert y_cal[g_cal].sum() >= 30, f"{name}: too few positives in calibration split ({y_cal[g_cal].sum()})"
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(
        s_cal[g_cal], y_cal[g_cal])

    s_te = anomaly_score(pipe, x_te)
    p_te = iso.predict(s_te)
    y_te = test["anomaly_type"].isin(explains).to_numpy()
    flagged = (p_te >= DECISION_P) & g_te

    m = {
        "n_test": int(len(test)), "positives": int(y_te.sum()),
        "calibrated_p_ge_0.5": prf(y_te, flagged),
        "baseline_iforest_default_threshold": prf(y_te, pipe.predict(x_te) == -1),
        "roc_auc": round(float(roc_auc_score(y_te, s_te)), 4),
        "pr_auc": round(float(average_precision_score(y_te, s_te)), 4),
        "recall_by_label": {},
        "false_positive_rate": round(float(flagged[~y_te].mean()), 4),
    }
    for lab in sorted(explains):
        mask = (test["anomaly_type"] == lab).to_numpy()
        m["recall_by_label"][lab] = {"n": int(mask.sum()),
                                     "recall": round(float(flagged[mask].mean()), 4) if mask.any() else None}
    return pipe, iso, m


def main() -> None:
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.TXN_CSV, parse_dates=["timestamp"]).sort_values("timestamp", kind="stable")
    plazas = pd.read_csv(cfg.PLAZA_CSV)
    df = df.reset_index(drop=True)

    n = len(df)
    i1, i2 = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + CAL_FRAC))
    train, cal, test = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    print(f"rows: train {len(train):,} | calibration {len(cal):,} | test {len(test):,}")
    print(f"test window starts {test['timestamp'].min()}")

    specs = {
        "api_flood": (cfg.flood_features, cfg.above_baseline, cfg.FLOOD_MODEL),
        "fare_anomaly": (lambda d: cfg.fare_features(d, plazas), None, cfg.FARE_MODEL),
    }
    calibrators, metrics = {}, {}
    for name, (featurize, gate, path) in specs.items():
        pipe, iso, m = run_detector(name, featurize, gate, cfg.DETECTOR_EXPLAINS[name], train, cal, test)
        joblib.dump(pipe, path)
        calibrators[name] = iso
        metrics[name] = m
        c = m["calibrated_p_ge_0.5"]
        b = m["baseline_iforest_default_threshold"]
        print(f"\n[{name}]  test positives: {m['positives']}")
        print(f"  calibrated (p>=0.5): P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}  "
              f"FP={c['fp']}  FPR={m['false_positive_rate']:.4f}")
        print(f"  default IF threshold: P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}")
        print(f"  ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}")
        for lab, v in m["recall_by_label"].items():
            print(f"  recall[{lab}] = {v['recall']}  (n={v['n']})")

    # Rule detectors have no fitted parameters (thresholds are physical limits), so they
    # are evaluated on the full dataset. Precision is Laplace-smoothed so a rule is never
    # assigned probability exactly 1.0 from a finite sample.
    rule_precision, rule_metrics = {}, {}
    for name, hit in ad.rule_flags(df, plazas).items():
        truth = set(df.loc[df["anomaly_type"].isin(cfg.DETECTOR_EXPLAINS[name]), "txn_id"])
        flagged = set(hit["txn_id"])
        tp = len(flagged & truth)
        rule_precision[name] = round((tp + 1) / (len(flagged) + 2), 4)
        rule_metrics[name] = {"flagged": len(flagged), "labelled": len(truth), "true_positives": tp,
                              "precision": round(tp / len(flagged), 4) if flagged else 0.0,
                              "recall": round(tp / len(truth), 4) if truth else None,
                              "probability_used": rule_precision[name]}
        print(f"\n[{name}] (rule) flagged {len(flagged)}  labelled {len(truth)}  TP {tp}  "
              f"-> probability used {rule_precision[name]}")
    metrics["rules"] = rule_metrics

    joblib.dump({"calibrators": calibrators, "decision_p": DECISION_P,
                 "rule_precision": rule_precision}, cfg.CALIBRATION)
    meta = {"sklearn": sklearn.__version__, "seed": cfg.SEED,
            "split": {"train": len(train), "calibration": len(cal), "test": len(test),
                      "test_starts": str(test["timestamp"].min())}}
    cfg.METRICS_JSON.write_text(json.dumps({"meta": meta, "models": metrics}, indent=2))
    print(f"\nSaved models -> {cfg.MODEL_DIR}")


if __name__ == "__main__":
    main()