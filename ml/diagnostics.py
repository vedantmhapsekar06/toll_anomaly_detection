"""Overfitting / underfitting diagnostics for the two ML detectors.

Run from the project root:  python ml/diagnostics.py   (after train.py; writes ml/models/diagnostics.json)

Seven checks. Five decide the verdict (1, 3, 4, 5, 7), each with a pass/fail rule decided BEFORE looking at
the numbers; two (2, 6) show stability over data size and time:

  1. train vs test gap          overfit if PR-AUC or F1 on the fit rows exceeds the test result by > 0.05
  2. learning curve             fit on 10%..100% of the data: train and validation scores must converge
  3. capacity curves            sweep trees and subsample size: too little capacity underfits, and the
                                chosen setting must sit on the plateau, not the edge
  4. headroom                   a higher-capacity supervised model on the same features: if it is far better
                                (> 0.05 PR-AUC) the label-free model is underfitting
  5. beats the baselines        test F1 must exceed the best simple non-ML rule
  6. rolling-origin folds       refit + recalibrate on 6 expanding windows; scores must be stable over time
  7. fresh worlds               5 newly generated datasets (new vehicles, trips, anomalies) scored with the
                                FINAL models, never refitted: F1 must stay within 0.05 of test

Also reported: block-bootstrap 95% confidence intervals and a calibration (reliability) check.
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

import common as C
import model_zoo as Z
from common import cfg
from data_loader import load_plazas, load_transactions
from generate_data import generate
from backend import anomaly_detector as ad

GAP_LIMIT = 0.05          # train-test gap that counts as overfitting
HEADROOM_LIMIT = 0.05     # supervised-reference headroom that counts as underfitting
PLATEAU_TOL = 0.01        # capacity curve must be within this of its best
FRESH_SEEDS = [101, 102, 103, 104, 105]
LEARNING_FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
N_BOOT, BLOCK = 2000, 250


def msd(v):
    v = np.asarray(v, float)
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4), "values": [round(float(x), 4) for x in v]}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


OVERFIT_CHECKS = ("train_test_gap", "fresh_world_f1")
UNDERFIT_CHECKS = ("headroom", "beats_baselines", "capacity_saturated")


def judge(checks: dict) -> tuple[str, dict]:
    """Overfit is tested first: a large train-test gap already explains a poor test result."""
    over = any(not checks[k]["pass"] for k in OVERFIT_CHECKS if k in checks)
    under = any(not checks[k]["pass"] for k in UNDERFIT_CHECKS if k in checks)
    return ("overfit" if over else "underfit" if under else "good fit"), {"overfit": over, "underfit": under}


def scores(spec, model, iso, df):
    return C.evaluate_split(spec, model, iso, df)


def main() -> None:
    warnings.filterwarnings("ignore")
    t_start = time.time()
    plazas = load_plazas()
    df = load_transactions(plazas=plazas)
    sp = C.split_time(df)
    train, cal, test = sp["train"], sp["calibration"], sp["test"]
    selection = json.loads(cfg.SELECTION_JSON.read_text())
    metrics = json.loads(cfg.METRICS_JSON.read_text())
    calib = joblib.load(cfg.CALIBRATION)
    specs = C.detector_specs(plazas)
    folds = C.selection_folds(train)
    pool, val_ref = folds[-1]          # fit on the first 6 chunks, validate on chunk 7 of the train window

    out = {"rules": {
        "overfit": f"train-minus-test gap > {GAP_LIMIT} in PR-AUC or F1, or fresh-world F1 more than {GAP_LIMIT} below test F1",
        "underfit": f"supervised headroom > {HEADROOM_LIMIT} PR-AUC, test F1 not above the best simple baseline, "
                    f"or capacity not saturated (largest setting more than {PLATEAU_TOL} better than the chosen one)",
        "good_fit": "none of the above"}, "detectors": {}}

    for name, spec in specs.items():
        print(f"\n================ {name} ================")
        model = joblib.load(spec["model_path"])
        iso = calib["calibrators"][name]
        chosen = selection["detectors"][name]["selected"]
        fam, params = chosen["family"], dict(chosen["params"])
        d = {"model": chosen["id"]}
        checks = {}

        # ---- 1. train vs test ---------------------------------------------------------
        tr, te = scores(spec, model, iso, train), scores(spec, model, iso, test)
        folds_info = selection["detectors"][name]["candidates"]
        sel_row = next(r for r in folds_info if r["id"] == chosen["id"])
        d["train_vs_test"] = {
            "train": tr, "test": te,
            "validation_folds": {"pr_auc_mean": sel_row["val_pr_auc"], "pr_auc_sd": sel_row["val_pr_auc_sd"],
                                 "fold_pr_auc": sel_row["fold_val_pr_auc"]},
            "gap_pr_auc": round(tr["pr_auc"] - te["pr_auc"], 4), "gap_f1": round(tr["f1"] - te["f1"], 4)}
        checks["train_test_gap"] = {"value": max(d["train_vs_test"]["gap_pr_auc"], d["train_vs_test"]["gap_f1"]),
                                    "limit": GAP_LIMIT, "pass": max(d["train_vs_test"]["gap_pr_auc"], d["train_vs_test"]["gap_f1"]) <= GAP_LIMIT}
        print(f"[1] train PR-AUC {tr['pr_auc']:.3f} / F1 {tr['f1']:.3f}  vs  test PR-AUC {te['pr_auc']:.3f} / F1 {te['f1']:.3f}"
              f"  -> gaps {d['train_vs_test']['gap_pr_auc']:+.3f} / {d['train_vs_test']['gap_f1']:+.3f}")

        # ---- 2. learning curve ----------------------------------------------------------
        X_pool, X_val = spec["featurize"](pool), spec["featurize"](val_ref)
        y_pool, y_val = C.labels(pool, spec), C.labels(val_ref, spec)
        lc = []
        for frac in LEARNING_FRACS:
            tr_s, va_s = [], []
            for seed in (0, 1, 2):
                idx = np.sort(np.random.default_rng(seed).choice(len(X_pool), int(len(X_pool) * frac), replace=False)) if frac < 1 else np.arange(len(X_pool))
                m = Z.build(fam, params)
                m.fit(X_pool.iloc[idx], y_pool[idx] if m.supervised else None, gate_col=spec["gate_col"])
                tr_s.append(C.ranking_metrics(y_pool[idx], m.score(X_pool.iloc[idx]), C.gate_mask(spec, m, X_pool.iloc[idx]))["pr_auc"])
                va_s.append(C.ranking_metrics(y_val, m.score(X_val), C.gate_mask(spec, m, X_val))["pr_auc"])
            lc.append({"fraction": frac, "rows": int(len(X_pool) * frac), "train_pr_auc": msd(tr_s)["mean"],
                       "val_pr_auc": msd(va_s)["mean"], "val_sd": msd(va_s)["sd"]})
        d["learning_curve"] = lc
        d["learning_curve_gap_at_full"] = round(lc[-1]["train_pr_auc"] - lc[-1]["val_pr_auc"], 4)
        print("[2] learning curve (train / validation PR-AUC): " +
              "  ".join(f"{int(r['fraction']*100)}%: {r['train_pr_auc']:.3f}/{r['val_pr_auc']:.3f}" for r in lc))

        # ---- 3. capacity curves ---------------------------------------------------------
        cap = {}
        if fam == "isolation_forest":
            sweeps = {"n_estimators": [10, 25, 50, 100, 200, 300, 600],
                      "max_samples": [16, 32, 64, 128, 256, 512, 1024, 2048]}
            for knob, values in sweeps.items():
                rows = []
                for v in values:
                    p2 = {**params, knob: v}
                    m = Z.build(fam, p2).fit(X_pool, None, gate_col=spec["gate_col"])
                    rows.append({"value": v,
                                 "train_pr_auc": C.ranking_metrics(y_pool, m.score(X_pool), C.gate_mask(spec, m, X_pool))["pr_auc"],
                                 "val_pr_auc": C.ranking_metrics(y_val, m.score(X_val), C.gate_mask(spec, m, X_val))["pr_auc"]})
                cap[knob] = {"chosen": params[knob], "curve": rows}
        d["capacity_curves"] = cap
        sat = {}
        for knob, c in cap.items():
            best = max(r["val_pr_auc"] for r in c["curve"])
            at = next(r["val_pr_auc"] for r in c["curve"] if r["value"] == c["chosen"])
            sat[knob] = {"best_val": round(best, 4), "chosen_val": round(at, 4), "shortfall": round(best - at, 4)}
        # Underfitting test = "would MORE capacity help?": the best validation score among tree counts
        # larger than the chosen one must not beat the chosen one by more than PLATEAU_TOL.
        # (The subsample-size curve is a sensitivity report, not an underfitting test.)
        shortfall = 0.0
        if "n_estimators" in cap:
            c = cap["n_estimators"]
            at = next(r["val_pr_auc"] for r in c["curve"] if r["value"] == c["chosen"])
            more = [r["val_pr_auc"] for r in c["curve"] if r["value"] > c["chosen"]]
            shortfall = max(0.0, max(more) - at) if more else 0.0
        d["capacity_saturation"] = sat
        print("[3] capacity: " + "; ".join(f"{k}: chosen {c['chosen']} val {sat[k]['chosen_val']:.3f}, best in sweep {sat[k]['best_val']:.3f}" for k, c in cap.items()))

        # ---- 4. headroom vs supervised reference ----------------------------------------
        sup = selection["detectors"][name]["best_supervised_reference"]
        headroom = selection["detectors"][name]["headroom_vs_supervised_pr_auc"]
        d["headroom"] = {"supervised_reference": sup["id"], "supervised_val_pr_auc": sup["val_pr_auc"],
                         "selected_val_pr_auc": chosen["val_pr_auc"], "headroom_pr_auc": headroom}
        print(f"[4] headroom vs supervised {sup['id']}: {headroom:+.3f} PR-AUC")

        # ---- 5. beats the baselines -----------------------------------------------------
        m_te = metrics["models"][name]
        base_f1 = max(m_te["baseline_simple_rule"]["f1"], m_te.get("baseline_context_rule", {"f1": 0})["f1"])
        d["baselines"] = {"best_baseline_f1": base_f1, "model_f1": m_te["calibrated_p_ge_0.5"]["f1"]}
        print(f"[5] test F1 {m_te['calibrated_p_ge_0.5']['f1']:.3f} vs best simple baseline {base_f1:.3f}")

        # ---- 6. rolling-origin folds (refit + recalibrate each time) --------------------
        chunks = np.array_split(np.arange(len(df)), 10)
        roll = []
        for j in range(4, 10):
            tr_j = df.iloc[:chunks[j - 1][0]]
            ca_j, te_j = df.iloc[chunks[j - 1][0]:chunks[j][0]], df.iloc[chunks[j][0]:chunks[j][-1] + 1]
            m_j, _ = C.fit_model(spec, lambda: Z.build(fam, params), tr_j)
            iso_j, _ = C.calibrate(spec, m_j, ca_j)
            r = C.evaluate_split(spec, m_j, iso_j, te_j)
            roll.append({"test_chunk": j, "train_rows": len(tr_j), "test_starts": str(te_j["timestamp"].min()),
                         "precision": r["precision"], "recall": r["recall"], "f1": r["f1"], "pr_auc": r["pr_auc"]})
        d["rolling_origin"] = {"folds": roll, **{k: msd([r[k] for r in roll]) for k in ("precision", "recall", "f1", "pr_auc")}}
        print(f"[6] rolling-origin F1 {d['rolling_origin']['f1']['mean']:.3f} +- {d['rolling_origin']['f1']['sd']:.3f} "
              f"(min {d['rolling_origin']['f1']['min']:.3f}, max {d['rolling_origin']['f1']['max']:.3f}) over {len(roll)} windows")

        # ---- bootstrap CI (moving blocks keep each burst of consecutive rows together) --
        pred = C.predict(spec, model, iso, test)
        y_te = C.labels(test, spec)
        n_blocks = len(test) // BLOCK
        tp_b = np.array([(y_te[i*BLOCK:(i+1)*BLOCK] & pred["flagged"][i*BLOCK:(i+1)*BLOCK]).sum() for i in range(n_blocks)])
        fp_b = np.array([(~y_te[i*BLOCK:(i+1)*BLOCK] & pred["flagged"][i*BLOCK:(i+1)*BLOCK]).sum() for i in range(n_blocks)])
        fn_b = np.array([(y_te[i*BLOCK:(i+1)*BLOCK] & ~pred["flagged"][i*BLOCK:(i+1)*BLOCK]).sum() for i in range(n_blocks)])
        rng = np.random.default_rng(7)
        pick = rng.integers(0, n_blocks, (N_BOOT, n_blocks))
        TP, FP, FN = tp_b[pick].sum(1), fp_b[pick].sum(1), fn_b[pick].sum(1)
        P, R = TP / np.maximum(TP + FP, 1), TP / np.maximum(TP + FN, 1)
        F = 2 * P * R / np.maximum(P + R, 1e-9)
        ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
        d["bootstrap_95ci"] = {"precision": ci(P), "recall": ci(R), "f1": ci(F), "blocks": n_blocks, "block_rows": BLOCK}
        print(f"    bootstrap 95% CI: precision {ci(P)}  recall {ci(R)}  F1 {ci(F)}")

        # ---- calibration check ---------------------------------------------------------
        g = pred["gate"]
        p_g, y_g = pred["p"][g], y_te[g]
        bins = np.linspace(0, 1, 11)
        which = np.clip(np.digitize(p_g, bins) - 1, 0, 9)
        rel = [{"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}", "n": int((which == b).sum()),
                "predicted": round(float(p_g[which == b].mean()), 4), "observed": round(float(y_g[which == b].mean()), 4)}
               for b in range(10) if (which == b).sum() >= 5]
        ece = float(sum(r["n"] * abs(r["predicted"] - r["observed"]) for r in rel) / max(sum(r["n"] for r in rel), 1))
        d["calibration"] = {"reliability": rel, "ece": round(ece, 4), "brier": round(float(np.mean((p_g - y_g) ** 2)), 4)}
        print(f"    calibration: ECE {ece:.3f}, Brier {d['calibration']['brier']:.4f}")

        out["detectors"][name] = {"data": d, "checks": checks, "_shortfall": shortfall}

    # ---- 7. fresh worlds -------------------------------------------------------------------
    print("\n================ fresh worlds (never seen, never refitted) ================")
    det = ad.TollGuardDetector()
    fresh = {}
    for seed in FRESH_SEEDS:
        new, _ = generate(seed, verbose=False)
        new = new.sort_values("timestamp", kind="stable").reset_index(drop=True)
        ev = det.analyze(new)["evaluation"]
        row = {k: {"precision": v["precision"], "recall": v["recall"],
                   "f1": round(2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"]), 4)
                   if v["precision"] and v["recall"] else 0.0} for k, v in ev.items()}
        for name, spec in specs.items():
            model, iso = joblib.load(spec["model_path"]), calib["calibrators"][name]
            row[name]["pr_auc"] = C.evaluate_split(spec, model, iso, new)["pr_auc"]
        fresh[seed] = row
        print(f"  seed {seed}: " + "  ".join(f"{k} F1 {v['f1']:.3f}" for k, v in row.items()))
    summary = {}
    for k in fresh[FRESH_SEEDS[0]]:
        summary[k] = {m_: msd([fresh[s][k][m_] for s in FRESH_SEEDS]) for m_ in ("precision", "recall", "f1")}
        if k in specs:
            summary[k]["pr_auc"] = msd([fresh[s][k]["pr_auc"] for s in FRESH_SEEDS])
    out["fresh_worlds"] = {"seeds": FRESH_SEEDS, "per_seed": fresh, "summary": summary}

    # ---- verdicts ------------------------------------------------------------------------
    print("\n================ VERDICTS ================")
    for name in specs:
        e = out["detectors"][name]
        d, checks = e["data"], e["checks"]
        test_f1 = d["train_vs_test"]["test"]["f1"]
        fresh_f1 = summary[name]["f1"]["mean"]
        checks["fresh_world_f1"] = {"value": round(test_f1 - fresh_f1, 4), "limit": GAP_LIMIT, "pass": (test_f1 - fresh_f1) <= GAP_LIMIT}
        checks["headroom"] = {"value": d["headroom"]["headroom_pr_auc"], "limit": HEADROOM_LIMIT,
                              "pass": d["headroom"]["headroom_pr_auc"] <= HEADROOM_LIMIT}
        checks["beats_baselines"] = {"value": round(d["baselines"]["model_f1"] - d["baselines"]["best_baseline_f1"], 4),
                                     "limit": 0.0, "pass": d["baselines"]["model_f1"] > d["baselines"]["best_baseline_f1"]}
        shortfall = e.pop("_shortfall")
        checks["capacity_saturated"] = {"value": round(shortfall, 4), "limit": PLATEAU_TOL, "pass": shortfall <= PLATEAU_TOL}
        e["verdict"], e["verdict_detail"] = judge(checks)
        print(f"{name}: {e['verdict'].upper()}   " + "  ".join(f"{k}={'ok' if v['pass'] else 'FAIL'}({v['value']})" for k, v in checks.items()))

    # ---- positive controls: can these checks actually FAIL? ---------------------------------
    print("\n================ positive controls (deliberately broken models) ================")
    codes = {p_: i for i, p_ in enumerate(sorted(plazas["plaza_id"]))}
    t0 = df["timestamp"].min()

    def memoriser(d):                         # knows only WHEN and WHERE, never WHAT happened
        return pd.DataFrame({"t": (d["timestamp"] - t0).dt.total_seconds(),
                             "plaza": d["plaza_id"].map(codes)}, index=d.index)

    out["controls"] = {}
    for name, spec in specs.items():
        first = list(spec["featurize"](train.iloc[:50]).columns)[0]
        defs = {
            "overfit_control": ("Gradient boosting on time + plaza only (memorises when and where anomalies happened)",
                                {**spec, "featurize": memoriser, "gate_col": None},
                                lambda: Z.build("gradient_boosting", dict(max_iter=400)), "overfit"),
            "underfit_control": (f"Isolation Forest on '{first}' alone (starved of the informative features)",
                                 {**spec, "featurize": (lambda d, f=spec["featurize"], o=first: f(d)[[o]]),
                                  "gate_col": spec["gate_col"] if spec["gate_col"] == first else None},
                                 lambda: Z.build("isolation_forest", dict(n_estimators=100, max_samples=256)), "underfit"),
        }
        sup_val = selection["detectors"][name]["best_supervised_reference"]["val_pr_auc"]
        base_f1 = out["detectors"][name]["data"]["baselines"]["best_baseline_f1"]
        out["controls"][name] = {}
        for key, (desc, sp2, builder, expected) in defs.items():
            m_c, _ = C.fit_model(sp2, builder, train)
            iso_c, _ = C.calibrate(sp2, m_c, cal)
            a, b = C.evaluate_split(sp2, m_c, iso_c, train), C.evaluate_split(sp2, m_c, iso_c, test)
            vals = []
            for f_df, v_df in folds:
                mm, _ = C.fit_model(sp2, builder, f_df)
                Xv = sp2["featurize"](v_df)
                vals.append(C.ranking_metrics(C.labels(v_df, sp2), mm.score(Xv), C.gate_mask(sp2, mm, Xv))["pr_auc"])
            head = round(sup_val - float(np.mean(vals)), 4)
            gap = max(round(a["pr_auc"] - b["pr_auc"], 4), round(a["f1"] - b["f1"], 4))
            ch = {"train_test_gap": {"value": gap, "limit": GAP_LIMIT, "pass": gap <= GAP_LIMIT},
                  "headroom": {"value": head, "limit": HEADROOM_LIMIT, "pass": head <= HEADROOM_LIMIT},
                  "beats_baselines": {"value": round(b["f1"] - base_f1, 4), "limit": 0.0, "pass": b["f1"] > base_f1}}
            verdict, _ = judge(ch)
            out["controls"][name][key] = {"description": desc, "train": a, "test": b, "checks": ch,
                                          "verdict": verdict, "expected": expected, "detected": verdict == expected}
            print(f"{name:<13}{key:<17} train PR-AUC {a['pr_auc']:.3f} / test {b['pr_auc']:.3f}  gap {gap:+.3f}  "
                  f"headroom {head:+.3f}  -> {verdict.upper()} (expected {expected}) "
                  f"{'DETECTED' if verdict == expected else 'MISSED'}")

    out["elapsed_seconds"] = round(time.time() - t_start, 1)
    cfg.DIAGNOSTICS_JSON.write_text(json.dumps(clean(out), indent=2))
    print(f"\nsaved -> {cfg.DIAGNOSTICS_JSON}  ({out['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
