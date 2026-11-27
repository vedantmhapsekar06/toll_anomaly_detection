"""Final training: fit the SELECTED model, calibrate it, evaluate it once on the test window.

Run from the project root:  python ml/train.py      (after ml/select_models.py)

    first 70% of time -> fit the selected model (labels NOT used by the deployable models)
    next  10%         -> fit the score calibrator (labels used)
    last  20%         -> test, read once, after every choice is locked

Every ML detector is also compared with simple non-ML baselines that get the same labelled
calibration data. If a detector cannot beat them, the report shows it.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import common as C
import model_zoo as Z
from common import cfg
from data_loader import load_plazas, load_transactions
from backend import anomaly_detector as ad  # noqa: E402

SENSITIVITY_SPEEDS = [90, 100, 120, 140, 160, 200, 300]
prf = C.prf


def tag_level(truth_tags, flagged_tags):
    tp = len(truth_tags & flagged_tags)
    return {"flagged": len(flagged_tags), "labelled": len(truth_tags), "true_positives": tp,
            "precision": round(tp / len(flagged_tags), 4) if flagged_tags else 0.0,
            "recall": round(tp / len(truth_tags), 4) if truth_tags else None}


def evaluate_rules(df, plazas):
    """Rules have no fitted parameters (their thresholds are physical limits or definitions), so
    they are evaluated on all rows. The labels come from the generator, not from the rules."""
    hits = ad.rule_flags(df, plazas)
    rule_precision, rule_metrics, detail = {}, {}, {}

    # cloned_tag: per transaction, with the ambiguous pairs reported separately
    h = hits["cloned_tag"]
    truth = set(df.loc[df["anomaly_type"] == "cloned_tag", "txn_id"])
    amb = set(h.loc[h["ambiguous"].astype(bool), "txn_id"])
    flagged = set(h["txn_id"])
    for key, ids in (("cloned_tag", flagged - amb), ("cloned_tag_ambiguous", amb)):
        tp = len(ids & truth)
        rule_precision[key] = round((tp + 1) / (len(ids) + 2), 4)      # Laplace smoothing
        detail[key] = {"flagged": len(ids), "true_positives": tp,
                       "precision": round(tp / len(ids), 4) if ids else None}
    tp = len(flagged & truth)
    rule_metrics["cloned_tag"] = {
        "flagged": len(flagged), "labelled": len(truth), "true_positives": tp,
        "precision": round(tp / len(flagged), 4) if flagged else 0.0,
        "recall": round(tp / len(truth), 4) if truth else None,
        "probability_used": rule_precision["cloned_tag"]}
    # how many genuine passages get blamed, and how many clones are caught at tag level
    clone_tags = set(df.loc[df["anomaly_type"] == "cloned_tag", "tag_id"])
    flagged_tags = set(df.loc[df["txn_id"].isin(flagged), "tag_id"])
    detail["cloned_tag_tag_level"] = tag_level(clone_tags, flagged_tags)

    # class_mismatch: per TAG (one alert per tag)
    h = hits["class_mismatch"]
    truth_t = set(df.loc[df["anomaly_type"] == "class_mismatch", "tag_id"])
    flag_t = set(df.loc[df["txn_id"].isin(h["txn_id"]), "tag_id"])
    r = tag_level(truth_t, flag_t)
    rule_precision["class_mismatch"] = round((r["true_positives"] + 1) / (r["flagged"] + 2), 4)
    rule_metrics["class_mismatch"] = {**r, "probability_used": rule_precision["class_mismatch"]}

    # duplicate_charge: per transaction
    h = hits["duplicate_charge"]
    truth = set(df.loc[df["anomaly_type"] == "duplicate_charge", "txn_id"])
    flagged = set(h["txn_id"])
    tp = len(flagged & truth)
    rule_precision["duplicate_charge"] = round((tp + 1) / (len(flagged) + 2), 4)
    rule_metrics["duplicate_charge"] = {
        "flagged": len(flagged), "labelled": len(truth), "true_positives": tp,
        "precision": round(tp / len(flagged), 4) if flagged else 0.0,
        "recall": round(tp / len(truth), 4) if truth else None,
        "probability_used": rule_precision["duplicate_charge"]}
    return rule_precision, rule_metrics, detail


def speed_sensitivity(df, plazas):
    """Precision/recall of the impossible-travel rule across speed limits."""
    truth = set(df.loc[df["anomaly_type"] == "cloned_tag", "txn_id"])
    out = {}
    for v in SENSITIVITY_SPEEDS:
        flagged = set(ad.detect_cloned_tag(df, plazas, max_speed=v)["txn_id"])
        tp = len(flagged & truth)
        out[str(v)] = {"flagged": len(flagged),
                       "precision": round(tp / len(flagged), 4) if flagged else None,
                       "recall": round(tp / len(truth), 4)}
    return out



def default_iforest_baseline(spec, train, test, y_te):
    """What you get from an untuned Isolation Forest using its built-in threshold."""
    pipe = Pipeline([("s", StandardScaler()), ("f", IsolationForest(n_estimators=100, contamination="auto",
                                                                    random_state=cfg.SEED, n_jobs=-1))])
    pipe.fit(spec["featurize"](train))
    return prf(y_te, pipe.predict(spec["featurize"](test)) == -1)


def main() -> None:
    warnings.filterwarnings("ignore")
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plazas = load_plazas()
    df = load_transactions(plazas=plazas)
    sp = C.split_time(df)
    train, cal, test = sp["train"], sp["calibration"], sp["test"]
    print(f"rows: train {len(train):,} | calibration {len(cal):,} | test {len(test):,}")
    print(f"test window starts {test['timestamp'].min()}")
    selection = json.loads(cfg.SELECTION_JSON.read_text())

    calibrators, metrics, prevalence, chosen_models = {}, {}, {}, {}
    for name, spec in C.detector_specs(plazas).items():
        chosen = selection["detectors"][name]["selected"]
        model, _ = C.fit_model(spec, lambda: Z.build(chosen["family"], chosen["params"]), train)
        iso, prev = C.calibrate(spec, model, cal)
        joblib.dump(model, spec["model_path"])
        calibrators[name], prevalence[name] = iso, prev
        chosen_models[name] = {"id": chosen["id"], "family": chosen["family"], "params": chosen["params"]}

        # ---- the one and only look at the test window ----
        pred = C.predict(spec, model, iso, test)
        y_cal, y_te = C.labels(cal, spec), C.labels(test, spec)
        main_m = prf(y_te, pred["flagged"])
        rank = C.ranking_metrics(y_te, pred["score"], pred["gate"])
        m = {
            "selected_model": chosen_models[name],
            "n_test": int(len(test)), "positives": int(y_te.sum()),
            "calibrated_p_ge_0.5": main_m,
            "baseline_iforest_default_threshold": default_iforest_baseline(spec, train, test, y_te),
            "baseline_simple_rule": C.BASELINES[name](cal, test, y_cal, y_te, plazas),
            "roc_auc": rank["roc_auc"], "pr_auc": rank["pr_auc"],
            "recall_by_label": {},
            "false_positive_rate": round(float(pred["flagged"][~y_te].mean()), 4),
            "precision_at_prevalence": {"1pct": C.precision_at_prevalence(main_m, 0.01),
                                        "0.1pct": C.precision_at_prevalence(main_m, 0.001)},
            "calibration_prevalence": round(prev, 4),
        }
        if name == "fare_anomaly":
            m["baseline_context_rule"] = C.fare_context_baseline(test, y_te, plazas)
        for lab in sorted(spec["explains"]):
            mask = (test["anomaly_type"] == lab).to_numpy()
            m["recall_by_label"][lab] = {"n": int(mask.sum()),
                                         "recall": round(float(pred["flagged"][mask].mean()), 4) if mask.any() else None}
        if spec["events"]:                                  # recall counted per EVENT, not per row
            ev = test.loc[y_te].assign(_flag=pred["flagged"][y_te]).groupby(spec["events"])["_flag"].mean()
            m["event_level"] = {"events": int(len(ev)), "detected": int((ev >= 0.5).sum()),
                                "recall": round(float((ev >= 0.5).mean()), 4) if len(ev) else None}
        metrics[name] = m

        c, b, s = m["calibrated_p_ge_0.5"], m["baseline_iforest_default_threshold"], m["baseline_simple_rule"]
        print(f"\n[{name}]  model: {chosen['id']}  | test positives: {m['positives']}")
        print(f"  calibrated (p>=0.5): P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}  FP={c['fp']}  "
              f"FPR={m['false_positive_rate']:.4f}")
        print(f"  default IF threshold: P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}")
        print(f"  simple baseline ({s['rule']}): P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")
        if "baseline_context_rule" in m:
            k = m["baseline_context_rule"]
            print(f"  context baseline: P={k['precision']:.3f} R={k['recall']:.3f} F1={k['f1']:.3f}")
        print(f"  ROC-AUC={m['roc_auc']}  PR-AUC={m['pr_auc']} (gated rows)  precision at 1% / 0.1% prevalence: "
              f"{m['precision_at_prevalence']['1pct']:.3f} / {m['precision_at_prevalence']['0.1pct']:.3f}")
        if "event_level" in m:
            e = m["event_level"]; print(f"  event level: {e['detected']}/{e['events']} floods detected")
        for lab, v in m["recall_by_label"].items():
            print(f"  recall[{lab}] = {v['recall']}  (n={v['n']})")

    rule_precision, rule_metrics, rule_detail = evaluate_rules(df, plazas)
    for k, v in rule_metrics.items():
        print(f"\n[{k}] (rule) flagged {v['flagged']}  labelled {v['labelled']}  TP {v['true_positives']}  "
              f"P={v['precision']}  R={v['recall']}  -> probability used {v['probability_used']}")
    metrics["rules"] = rule_metrics
    sensitivity = speed_sensitivity(df, plazas)
    print(f"\nspeed-limit sensitivity: {json.dumps(sensitivity)}")

    joblib.dump({"calibrators": calibrators, "decision_p": C.DECISION_P, "rule_precision": rule_precision,
                 "train_prevalence": prevalence}, cfg.CALIBRATION)
    meta = {"sklearn": sklearn.__version__, "seed": cfg.SEED,
            "split": {"train": len(train), "calibration": len(cal), "test": len(test),
                      "test_starts": str(test["timestamp"].min())},
            "selected_models": chosen_models,
            "anomaly_rate": round(float((df["anomaly_type"] != "none").mean()), 4)}
    cfg.METRICS_JSON.write_text(json.dumps(
        {"meta": meta, "models": metrics, "rule_detail": rule_detail,
         "rule_sensitivity": {"cloned_tag": sensitivity}}, indent=2))
    print(f"\nSaved models -> {cfg.MODEL_DIR}")


if __name__ == "__main__":
    main()
