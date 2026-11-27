"""Model selection: compare candidate models per detector with rolling-origin validation.

Run from the project root:  python ml/select_models.py     (writes ml/models/selection.json)

Protocol
--------
* The TRAIN window (first 70% of time) is cut into 7 chunks. Four folds fit a candidate on
  everything before a chunk and validate on that chunk, so a candidate is judged on four
  different stretches of time, not one.
* Candidates are ranked by mean validation PR-AUC (threshold-free, robust to the ~3% class
  balance), measured on the rows the detector is allowed to flag (its gate).
* Only UNSUPERVISED candidates can be deployed: real toll data has no labels until analysts
  confirm alerts. Supervised models are fitted for reference, to measure the headroom the
  label-free approach leaves.
* Winner = the CHEAPEST (fastest-fitting) deployable candidate that is not significantly worse
  than the best one. "Significantly worse" is a paired comparison across folds: its mean
  difference to the best must be more than one standard error below zero.
* Low-capacity ANCHOR settings are fitted only to draw the capacity curve; never selected.
* Calibration and test windows are never read here.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np

import common as C
import model_zoo as Z
from common import cfg
from data_loader import load_plazas, load_transactions


def mean_sd(v):
    v = np.asarray(v, float)
    return round(float(v.mean()), 4), round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)


def clean(o):                                           # NaN is not valid JSON
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return None if isinstance(o, float) and np.isnan(o) else o


def main() -> None:
    warnings.filterwarnings("ignore")
    plazas = load_plazas()
    df = load_transactions(plazas=plazas)
    splits = C.split_time(df)
    folds = C.selection_folds(splits["train"])
    specs = C.detector_specs(plazas)
    print("split rows:", {k: len(v) for k, v in splits.items()}, "| folds (fit rows -> val rows):",
          [(len(f), len(v)) for f, v in folds])

    out = {"protocol": {
        "selection_metric": "mean validation PR-AUC over 4 rolling-origin folds, on gated rows",
        "winner_rule": "cheapest deployable candidate not significantly worse than the best (paired, 1 SE)",
        "deployable": "unsupervised families only",
        "split_rows": {k: len(v) for k, v in splits.items()},
        "fold_rows": [{"fit": len(f), "validate": len(v)} for f, v in folds]}, "detectors": {}}

    for name, spec in specs.items():
        feats = [(spec["featurize"](f), spec["featurize"](v), C.labels(f, spec), C.labels(v, spec)) for f, v in folds]
        print(f"\n=== {name}: validation positives per fold {[int(y.sum()) for *_, y in feats]}")
        rows = []
        for family, params in Z.GRID:
            cid = Z.config_id(family, params)
            row = {"id": cid, "family": family, "params": params, "supervised": Z.is_supervised(family),
                   "anchor": Z.is_anchor(family, params), "capacity_knob": Z.capacity(family, params)[0],
                   "capacity": Z.capacity(family, params)[1]}
            tr_pr, va_pr, va_roc, secs = [], [], [], []
            try:
                for X_f, X_v, y_f, y_v in feats:
                    t0 = time.time()
                    m = Z.build(family, params)
                    m.fit(X_f, y_f if m.supervised else None, gate_col=spec["gate_col"])
                    secs.append(time.time() - t0)
                    tr_pr.append(C.ranking_metrics(y_f, m.score(X_f), C.gate_mask(spec, m, X_f))["pr_auc"])
                    va = C.ranking_metrics(y_v, m.score(X_v), C.gate_mask(spec, m, X_v))
                    va_pr.append(va["pr_auc"]); va_roc.append(va["roc_auc"])
                row.update(fold_val_pr_auc=[round(x, 4) for x in va_pr], fold_train_pr_auc=[round(x, 4) for x in tr_pr],
                           val_pr_auc=mean_sd(va_pr)[0], val_pr_auc_sd=mean_sd(va_pr)[1],
                           train_pr_auc=mean_sd(tr_pr)[0], val_roc_auc=mean_sd(va_roc)[0],
                           gap=round(mean_sd(tr_pr)[0] - mean_sd(va_pr)[0], 4), fit_seconds=round(float(np.mean(secs)), 2))
            except Exception as e:                       # a failing candidate is a result, not a crash
                row.update(error=f"{type(e).__name__}: {str(e)[:120]}", val_pr_auc=float("nan"))
            rows.append(row)
            print(f"  {cid:<60} train {row.get('train_pr_auc', float('nan')):.3f}  "
                  f"val {row.get('val_pr_auc', float('nan')):.3f} +-{row.get('val_pr_auc_sd', float('nan')):.3f}  "
                  f"{row.get('fit_seconds', 0):>5.1f}s {'ANCHOR' if row['anchor'] else ''} {row.get('error', '')}")

        ok = [r for r in rows if "error" not in r]
        eligible = [r for r in ok if not r["supervised"] and not r["anchor"]]
        best = max(eligible, key=lambda r: r["val_pr_auc"])
        contenders = [best]
        for r in eligible:
            if r is best:
                continue
            d = np.asarray(r["fold_val_pr_auc"]) - np.asarray(best["fold_val_pr_auc"])
            se = d.std(ddof=1) / np.sqrt(len(d))
            if d.mean() + se >= 0:                       # best is not significantly better
                contenders.append(r)
        chosen = min(contenders, key=lambda r: r["fit_seconds"])
        sup = max((r for r in ok if r["supervised"]), key=lambda r: r["val_pr_auc"])
        keys = ("id", "family", "params", "train_pr_auc", "val_pr_auc", "val_pr_auc_sd", "gap")
        out["detectors"][name] = {
            "candidates": rows,
            "best_deployable": {k: best[k] for k in keys},
            "indistinguishable_from_best": [r["id"] for r in contenders],
            "selected": {k: chosen[k] for k in keys},
            "best_supervised_reference": {k: sup[k] for k in keys},
            "headroom_vs_supervised_pr_auc": round(sup["val_pr_auc"] - chosen["val_pr_auc"], 4),
        }
        print(f"  best deployable: {best['id']} ({best['val_pr_auc']:.3f}); statistically tied: "
              f"{[r['id'] for r in contenders]}\n  -> SELECTED {chosen['id']}  | supervised reference "
              f"{sup['id']} val {sup['val_pr_auc']:.3f} | headroom {sup['val_pr_auc'] - chosen['val_pr_auc']:+.3f}")

    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.SELECTION_JSON.write_text(json.dumps(clean(out), indent=2))
    print(f"\nsaved -> {cfg.SELECTION_JSON}")


if __name__ == "__main__":
    main()
