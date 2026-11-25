"""TollGuard detection engine.

Three rule detectors (cloned_tag, class_mismatch, duplicate_charge) and two
calibrated Isolation Forests (fare_anomaly, api_flood). Every flag carries a
probability and a human-readable reason; flags on the same transaction are
combined with noisy-OR into one 0-100 risk score.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from backend import config as cfg

DETECTOR_ORDER = list(cfg.DETECTOR_EXPLAINS)   # cloned_tag, class_mismatch, duplicate_charge, fare_anomaly, api_flood

# ----------------------------------------------------------------------------
# 1. Column standardisation and validation
# ----------------------------------------------------------------------------
_ALIASES = {
    "tagid": "tag_id", "tag": "tag_id", "fastag_id": "tag_id", "fastag": "tag_id",
    "transaction_id": "txn_id", "txn": "txn_id", "id": "txn_id",
    "plaza": "plaza_id", "toll_plaza": "plaza_id", "plazaid": "plaza_id",
    "time": "timestamp", "datetime": "timestamp", "date_time": "timestamp",
    "vehicle_class": "declared_class", "class": "declared_class", "tag_class": "declared_class",
    "axles": "axle_count", "axle": "axle_count", "no_of_axles": "axle_count",
    "amount": "amount_INR", "fare": "amount_INR", "toll_amount": "amount_INR",
    "requests_per_min": "req_per_min", "req_rate": "req_per_min",
    "cpu": "cpu_usage_percent", "cpu_usage": "cpu_usage_percent", "cpu_percent": "cpu_usage_percent",
}


def _key(col) -> str:
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    canon = {_key(c): c for c in cfg.REQUIRED_COLUMNS + cfg.TELEMETRY_COLUMNS + cfg.OPTIONAL_COLUMNS}
    mapping = {}
    for col in df.columns:
        target = canon.get(_key(col)) or _ALIASES.get(_key(col))
        if target and target not in mapping.values():
            mapping[col] = target
    return df.rename(columns=mapping)


def prepare(df_raw: pd.DataFrame, plazas: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Standardise, coerce types, drop unusable rows. Returns (clean_df, info_messages)."""
    notes: list[str] = []
    df = standardize_columns(df_raw.copy())

    if "txn_id" not in df.columns:
        df.insert(0, "txn_id", [f"ROW{i:07d}" for i in range(1, len(df) + 1)])
        notes.append("No txn_id column found: row numbers were used as transaction IDs.")
    missing = [c for c in cfg.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}. Required: {cfg.REQUIRED_COLUMNS}")
    if df["txn_id"].astype(str).duplicated().any():
        raise ValueError("txn_id values must be unique.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["axle_count", "amount_INR", *[t for t in cfg.TELEMETRY_COLUMNS if t in df.columns]]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["declared_class"] = df["declared_class"].astype("string").str.strip().str.lower()

    bad = df[cfg.REQUIRED_COLUMNS].isna().any(axis=1) | ~df["declared_class"].isin(cfg.CLASSES)
    if bad.any():
        notes.append(f"Dropped {int(bad.sum())} row(s) with missing values or an unknown vehicle class.")
        df = df[~bad].copy()

    for c in ["txn_id", "tag_id", "plaza_id", "declared_class"]:
        df[c] = df[c].astype(str).str.strip()
    df["axle_count"] = df["axle_count"].astype(int)

    unknown = ~df["plaza_id"].isin(plazas["plaza_id"])
    if unknown.any():
        notes.append(f"{int(unknown.sum())} row(s) have a plaza_id not in plazas.csv: "
                     "cloned-tag and fare checks skip them.")
    return df.reset_index(drop=True), notes


# ----------------------------------------------------------------------------
# 2. Rule detectors (each returns a DataFrame with txn_id and reason)
# ----------------------------------------------------------------------------
def detect_cloned_tag(df: pd.DataFrame, plazas: pd.DataFrame) -> pd.DataFrame:
    """Same tag at two plazas faster than any vehicle could travel between them."""
    geo = plazas[["plaza_id", "name", "lat", "lon"]]
    d = df[["txn_id", "tag_id", "plaza_id", "timestamp"]].merge(geo, on="plaza_id", how="left")
    d = d.sort_values(["tag_id", "timestamp"], kind="stable")
    g = d.groupby("tag_id")
    d["dist_km"] = cfg.haversine_km(g["lat"].shift(), g["lon"].shift(), d["lat"], d["lon"])
    d["gap_h"] = (d["timestamp"] - g["timestamp"].shift()).dt.total_seconds() / 3600
    d["prev_plaza"] = g["name"].shift()
    d["speed"] = d["dist_km"] / d["gap_h"].clip(lower=1 / 3600)
    hit = d[d["speed"] > cfg.MAX_SPEED_KMH].copy()
    hit["reason"] = [
        f"Implied speed {s:.0f} km/h: {dk:.0f} km from {pp} to {n} in {gh * 60:.1f} min "
        f"(limit {cfg.MAX_SPEED_KMH} km/h)"
        for s, dk, pp, n, gh in zip(hit["speed"], hit["dist_km"], hit["prev_plaza"], hit["name"], hit["gap_h"])
    ]
    return hit[["txn_id", "reason"]]


def detect_class_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """Declared (tag) class differs from the class implied by the axle count."""
    true_cls = pd.Series(cfg.axle_to_class(df["axle_count"]), index=df.index)
    hit = df[true_cls != df["declared_class"]].copy()
    hit["reason"] = [
        f"Tag declares {dc} but {ax} axles = {tc}; billed INR {amt:.0f}"
        for dc, ax, tc, amt in zip(hit["declared_class"], hit["axle_count"], true_cls[hit.index], hit["amount_INR"])
    ]
    return hit[["txn_id", "reason"]]


def detect_duplicate(df: pd.DataFrame, plazas: pd.DataFrame) -> pd.DataFrame:
    """Same tag charged again at the same plaza within DUPLICATE_WINDOW_S seconds."""
    names = plazas.set_index("plaza_id")["name"]
    d = df[["txn_id", "tag_id", "plaza_id", "timestamp"]].sort_values(
        ["tag_id", "plaza_id", "timestamp"], kind="stable")
    g = d.groupby(["tag_id", "plaza_id"])
    d["gap_s"] = (d["timestamp"] - g["timestamp"].shift()).dt.total_seconds()
    d["prev_txn"] = g["txn_id"].shift()
    hit = d[d["gap_s"] < cfg.DUPLICATE_WINDOW_S].copy()
    hit["reason"] = [
        f"Charged again at {names.get(p, p)} {gs:.0f}s after {pt}"
        for p, gs, pt in zip(hit["plaza_id"], hit["gap_s"], hit["prev_txn"])
    ]
    return hit[["txn_id", "reason"]]


def rule_flags(df: pd.DataFrame, plazas: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All three rule detectors. Used by the live detector and by train.py for evaluation."""
    return {
        "cloned_tag": detect_cloned_tag(df, plazas),
        "class_mismatch": detect_class_mismatch(df),
        "duplicate_charge": detect_duplicate(df, plazas),
    }


def _noisy_or(s: pd.Series) -> float:
    """P(at least one signal is real) assuming independent signals."""
    return 1.0 - float(np.prod(1.0 - s.to_numpy(dtype=float)))


# ----------------------------------------------------------------------------
# 3. The detector (rules + calibrated ML + scoring + merge)
# ----------------------------------------------------------------------------
class TollGuardDetector:
    def __init__(self) -> None:
        self.plazas = pd.read_csv(cfg.PLAZA_CSV)
        self.models = {"api_flood": joblib.load(cfg.FLOOD_MODEL), "fare_anomaly": joblib.load(cfg.FARE_MODEL)}
        cal = joblib.load(cfg.CALIBRATION)
        self.calibrators = cal["calibrators"]
        self.decision_p = cal["decision_p"]
        self.rule_precision = cal["rule_precision"]

    # ---- ML detectors ----------------------------------------------------
    def _ml_flags(self, name: str, feats: pd.DataFrame, df: pd.DataFrame, reason_fn, gate=None) -> pd.DataFrame:
        ok = np.isfinite(feats.to_numpy()).all(axis=1)
        if not ok.any():
            return pd.DataFrame(columns=["txn_id", "detector", "p", "reason"])
        x = feats[ok]
        score = -self.models[name].score_samples(x)
        p = self.calibrators[name].predict(score)
        keep = p >= self.decision_p
        if gate is not None:
            keep &= gate(self.models[name], x)
        idx = x.index[keep]
        return pd.DataFrame({
            "txn_id": df.loc[idx, "txn_id"].to_numpy(), "detector": name, "p": p[keep],
            "reason": [reason_fn(df.loc[i], feats.loc[i]) for i in idx],
        })

    def detect_fare_anomaly(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = cfg.fare_features(df, self.plazas)

        def reason(row, f):
            true_cls = cfg.axle_to_class([row["axle_count"]])[0]
            return (f"Billed INR {row['amount_INR']:.0f} = {f['fare_ratio']:.2f}x expected for "
                    f"declared {row['declared_class']}, {f['axle_fare_ratio']:.2f}x for {true_cls} "
                    f"({row['axle_count']} axles)")
        return self._ml_flags("fare_anomaly", feats, df, reason)

    def detect_api_flood(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = cfg.flood_features(df)

        def reason(row, f):
            return (f"Gateway load {row['req_per_min']:.0f} req/min at {row['cpu_usage_percent']:.0f}% CPU "
                    f"(plaza {row['plaza_id']})")
        return self._ml_flags("api_flood", feats, df, reason, gate=cfg.above_baseline)

    # ---- full pipeline ---------------------------------------------------
    def collect_flags(self, df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
        parts = []
        for name, hit in rule_flags(df, self.plazas).items():
            hit = hit.assign(detector=name, p=self.rule_precision[name])
            parts.append(hit[["txn_id", "detector", "p", "reason"]])
        parts.append(self.detect_fare_anomaly(df))
        if all(c in df.columns for c in cfg.TELEMETRY_COLUMNS):
            parts.append(self.detect_api_flood(df))
        else:
            notes.append("req_per_min / cpu_usage_percent not found: api_flood check skipped.")
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame(columns=["txn_id", "detector", "p", "reason"])
        return pd.concat(parts, ignore_index=True)

    def merge_and_score(self, df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
        cols = ["txn_id", "tag_id", "plaza_id", "plaza_name", "timestamp", "declared_class",
                "axle_count", "amount_INR", "detected_type", "risk_score", "confidence", "severity",
                "n_flags", "reason"]
        if flags.empty:
            return pd.DataFrame(columns=cols)
        f = flags.copy()
        f["detector"] = pd.Categorical(f["detector"], categories=DETECTOR_ORDER, ordered=True)
        f["pw"] = f["p"].astype(float) * f["detector"].astype(str).map(cfg.IMPACT_WEIGHT)
        f = f.sort_values(["txn_id", "detector"], kind="stable")
        agg = f.groupby("txn_id", sort=False, observed=True).agg(
            risk=("pw", _noisy_or),            # probability x impact, combined across detectors
            conf=("p", _noisy_or),             # pure calibrated probability, combined
            detected_type=("detector", lambda s: "+".join(map(str, s))),
            reason=("reason", " | ".join),
            n_flags=("detector", "size"),
        ).reset_index()
        agg["risk_score"] = (agg["risk"] * 100).clip(upper=100).round(1)
        agg["confidence"] = (agg["conf"] * 100).clip(upper=100).round(1)
        agg["severity"] = np.where(agg["risk_score"] >= cfg.CRITICAL_SCORE, "Critical",
                                   np.where(agg["risk_score"] >= cfg.WARNING_SCORE, "Warning", "Info"))
        out = df.merge(agg, on="txn_id", how="inner")
        out["plaza_name"] = out["plaza_id"].map(self.plazas.set_index("plaza_id")["name"]).fillna(out["plaza_id"])
        return out.sort_values(["risk_score", "timestamp"], ascending=[False, True])[cols].reset_index(drop=True)

    def analyze(self, df_raw: pd.DataFrame) -> dict:
        df, notes = prepare(df_raw, self.plazas)
        flags = self.collect_flags(df, notes)
        results = self.merge_and_score(df, flags)
        out = {
            "results": results,
            "flags": flags,
            "notes": notes,
            "summary": self.summarize(df, flags, results),
        }
        if "anomaly_type" in df.columns:
            out["evaluation"] = self.evaluate(df, flags)
        return out

    # ---- reporting -------------------------------------------------------
    @staticmethod
    def summarize(df, flags, results) -> dict:
        return {
            "total_transactions": int(len(df)),
            "flagged_transactions": int(len(results)),
            "flag_rate": round(len(results) / max(len(df), 1), 4),
            "by_severity": {k: int(v) for k, v in results["severity"].value_counts().items()},
            "by_detector": {k: int(v) for k, v in flags["detector"].value_counts().items()} if len(flags) else {},
            "by_plaza": {k: int(v) for k, v in results["plaza_name"].value_counts().head(10).items()},
        }

    @staticmethod
    def evaluate(df: pd.DataFrame, flags: pd.DataFrame) -> dict:
        """Only when the upload carries an anomaly_type ground-truth column."""
        out = {}
        for det, labels in cfg.DETECTOR_EXPLAINS.items():
            truth = set(df.loc[df["anomaly_type"].isin(labels), "txn_id"])
            pred = set(flags.loc[flags["detector"] == det, "txn_id"]) if len(flags) else set()
            tp = len(truth & pred)
            out[det] = {"flagged": len(pred), "labelled": len(truth), "true_positives": tp,
                        "precision": round(tp / len(pred), 4) if pred else None,
                        "recall": round(tp / len(truth), 4) if truth else None}
        return out