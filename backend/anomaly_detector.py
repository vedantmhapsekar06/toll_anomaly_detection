"""TollGuard detection engine.

Three rule detectors (cloned_tag, class_mismatch, duplicate_charge) and two
calibrated Isolation Forests (fare_anomaly, api_flood). Every flag carries a
probability and a human-readable reason; flags on the same transaction are
combined with noisy-OR into one 0-100 risk score. Each detector reads a different
signal, so the independence that noisy-OR assumes is not violated by design.
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
_EMPTY_RULE = ["txn_id", "reason"]


def _dominant_corridor(corridors: list):
    """The corridor most of a tag's passages are on, only if it is a clear majority."""
    known = [c for c in corridors if isinstance(c, str)]
    if not known:
        return None
    top = max(set(known), key=known.count)
    return top if known.count(top) * 2 > len(corridors) else None


def _resolve_clone_conflicts(t: pd.DataFrame, max_speed: float, dominant) -> tuple:
    """Decide WHICH passages of one tag are the cloned ones.

    Two passages conflict when the speed needed to get from one plaza to the other in the
    time between them exceeds the limit. A cloned passage conflicts with every genuine
    passage near it in time, whereas a genuine passage conflicts only with clones, so the
    right answer is the smallest set of passages that removes every conflict (greedy
    vertex cover on the conflict graph). Flagging "the later passage of each bad pair"
    would blame the genuine vehicle whenever it passes AFTER the clone.

    When a lone pair is all there is, the data cannot say which is the clone. The passage
    that is off the tag's usual corridor is blamed if there is a clear usual corridor;
    otherwise both are flagged and marked ambiguous.

    Returns [(row position in t, ambiguous, position of the conflicting passage)].
    """
    lat = t["lat"].to_numpy(dtype=float)
    lon = t["lon"].to_numpy(dtype=float)
    secs = (t["timestamp"] - t["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    corr = t["corridor"].tolist()
    dist = cfg.haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    hours = np.abs(secs[:, None] - secs[None, :]) / 3600
    speed = dist / np.clip(hours, 1 / 3600, None)
    full = speed > max_speed                      # NaN (unknown plaza) compares False
    conflict = np.triu(full, k=1)                 # each pair once

    def off_route(i: int) -> bool:
        return dominant is not None and isinstance(corr[i], str) and corr[i] != dominant

    picked: dict[int, bool] = {}
    work = conflict.copy()
    while work.any():
        deg = work.sum(axis=0) + work.sum(axis=1)
        top = int(deg.max())
        if top == 1:                               # only isolated pairs remain
            for a, b in zip(*np.nonzero(work)):
                offs = [x for x in (int(a), int(b)) if off_route(x)]
                if len(offs) == 1:
                    picked.setdefault(offs[0], False)
                else:
                    picked.setdefault(int(a), True)
                    picked.setdefault(int(b), True)
            break
        cand = [int(x) for x in np.flatnonzero(deg == top)]
        if len(cand) > 1:
            cand = [x for x in cand if off_route(x)] or cand
        pick = cand[-1]                            # rows are time-ordered: latest of any tie
        picked.setdefault(pick, False)
        work[pick, :] = False
        work[:, pick] = False

    out = []
    for i, amb in picked.items():
        partners = np.flatnonzero(full[i, :])
        j = int(partners[np.argmax(speed[i, partners])])
        out.append((i, amb, j))
    return out, dist, hours, speed


def detect_cloned_tag(df: pd.DataFrame, plazas: pd.DataFrame, max_speed: float | None = None) -> pd.DataFrame:
    """Same tag at two plazas faster than any vehicle could travel between them.
    Returns txn_id, reason and `ambiguous` (True when the data cannot say which of the two
    passages is the clone; both are then flagged)."""
    max_speed = cfg.MAX_SPEED_KMH if max_speed is None else max_speed
    cols = _EMPTY_RULE + ["ambiguous"]
    geo = plazas[["plaza_id", "name", "lat", "lon", "corridor"]]
    d = df[["txn_id", "tag_id", "plaza_id", "timestamp"]].merge(geo, on="plaza_id", how="left")
    d = d.sort_values(["tag_id", "timestamp"], kind="stable").reset_index(drop=True)

    # Cheap vectorised pass: only tags with an impossible consecutive pair need the full analysis
    g = d.groupby("tag_id")
    dist = cfg.haversine_km(g["lat"].shift(), g["lon"].shift(), d["lat"], d["lon"])
    gap_h = (d["timestamp"] - g["timestamp"].shift()).dt.total_seconds() / 3600
    suspects = d.loc[dist / gap_h.clip(lower=1 / 3600) > max_speed, "tag_id"].unique()
    if len(suspects) == 0:
        return pd.DataFrame(columns=cols)

    rows = []
    for tag, t in d[d["tag_id"].isin(suspects)].groupby("tag_id", sort=False):
        t = t.reset_index(drop=True)
        dominant = _dominant_corridor(t["corridor"].tolist())
        picked, dk, hr, sp = _resolve_clone_conflicts(t, max_speed, dominant)
        for i, amb, j in picked:
            a, b = (i, j) if i < j else (j, i)           # describe the pair in time order
            reason = (f"Implied speed {sp[i, j]:.0f} km/h: {dk[i, j]:.0f} km from {t.at[a, 'name']} to "
                      f"{t.at[b, 'name']} in {hr[i, j] * 60:.1f} min (limit {max_speed:g} km/h)")
            reason += (f"; paired with {t.at[j, 'txn_id']}, cannot tell which passage is the clone"
                       if amb else f"; conflicts with {t.at[j, 'txn_id']}")
            rows.append((t.at[i, "txn_id"], reason, amb))
    return pd.DataFrame(rows, columns=cols)


def detect_class_mismatch(df: pd.DataFrame, plazas: pd.DataFrame) -> pd.DataFrame:
    """A tag whose declared class persistently differs from the class its axles imply.
    One alert per TAG (on its first mismatching passage in the file), not one per passage.
    A lone mismatch is more likely an axle-sensor misread, so it is not flagged."""
    true_cls = cfg.axle_to_class(df["axle_count"])
    factor = df["plaza_id"].map(plazas.set_index("plaza_id")["fare_factor"]).fillna(1.0).to_numpy(dtype=float)
    d = pd.DataFrame({
        "txn_id": df["txn_id"].to_numpy(), "tag_id": df["tag_id"].to_numpy(),
        "timestamp": df["timestamp"].to_numpy(), "declared": df["declared_class"].to_numpy(),
        "true": true_cls, "axles": df["axle_count"].to_numpy(), "amount": df["amount_INR"].to_numpy(dtype=float),
        "expected": cfg.expected_fare(true_cls, factor),
    })
    d["mis"] = d["true"] != d["declared"]
    stats = d.groupby("tag_id")["mis"].agg(n="size", n_mis="sum")
    bad = stats[(stats["n_mis"] >= cfg.CLASS_MISMATCH_MIN_PASSAGES)
                & (stats["n_mis"] / stats["n"] >= cfg.CLASS_MISMATCH_MIN_SHARE)]
    if bad.empty:
        return pd.DataFrame(columns=_EMPTY_RULE)

    m = d[d["tag_id"].isin(bad.index) & d["mis"]].sort_values(["tag_id", "timestamp"], kind="stable")
    first = m.groupby("tag_id", sort=False).head(1).set_index("tag_id")
    agg = m.groupby("tag_id").agg(billed=("amount", "sum"), expected=("expected", "sum"),
                                  true=("true", lambda s: s.mode().iat[0]),
                                  axles=("axles", lambda s: int(s.mode().iat[0])))
    rows = []
    for tag, r in first.iterrows():
        a, n = agg.loc[tag], bad.loc[tag]
        rows.append((r["txn_id"],
                     f"Tag declares {r['declared']} but {int(n['n_mis'])} of {int(n['n'])} passages show "
                     f"{a['axles']} axles = {a['true']}; billed INR {a['billed']:,.0f} vs INR "
                     f"{a['expected']:,.0f} expected for {a['true']} (first seen here)"))
    return pd.DataFrame(rows, columns=_EMPTY_RULE)


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
        "class_mismatch": detect_class_mismatch(df, plazas),
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
        self.train_prevalence = cal.get("train_prevalence", {})

    # ---- ML detectors ----------------------------------------------------
    def _ml_flags(self, name: str, feats: pd.DataFrame, df: pd.DataFrame, reason_fn, gate=None) -> pd.DataFrame:
        ok = np.isfinite(feats.to_numpy()).all(axis=1)
        if not ok.any():
            return pd.DataFrame(columns=["txn_id", "detector", "p", "reason"])
        x = feats[ok]
        score = self.models[name].score(x)
        p = self.calibrators[name].predict(score)
        deploy = cfg.DEPLOYMENT_PREVALENCE.get(name)
        if deploy and self.train_prevalence.get(name):
            p = cfg.shift_prior(p, self.train_prevalence[name], deploy)
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
            return (f"Billed INR {row['amount_INR']:.0f} = {f['fare_ratio']:.2f}x the tariff for declared "
                    f"{row['declared_class']} ({f['vs_tag']:.2f}x this tag's usual billing, "
                    f"{f['vs_plaza']:.2f}x the plaza's recent level)")
        return self._ml_flags("fare_anomaly", feats, df, reason)

    def detect_api_flood(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = cfg.flood_features(df)

        def reason(row, f):
            return (f"Gateway load {row['req_per_min']:.0f} req/min at {row['cpu_usage_percent']:.0f}% CPU "
                    f"(plaza {row['plaza_id']}; recent plaza level {np.expm1(f['ctx_req']):.0f} req/min)")
        return self._ml_flags("api_flood", feats, df, reason, gate=cfg.above_baseline)

    # ---- full pipeline ---------------------------------------------------
    def collect_flags(self, df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
        parts = []
        for name, hit in rule_flags(df, self.plazas).items():
            if hit.empty:
                continue
            if name == "cloned_tag":
                p = np.where(hit["ambiguous"].astype(bool),
                             self.rule_precision.get("cloned_tag_ambiguous", self.rule_precision[name]),
                             self.rule_precision[name])
            else:
                p = self.rule_precision[name]
            hit = hit.assign(detector=name, p=p)
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
        """Only when the upload carries an anomaly_type ground-truth column.
        class_mismatch is a tag-level detector, so it is scored per tag; the rest per transaction."""
        out = {}
        tag_of = df.set_index("txn_id")["tag_id"]
        for det, labels in cfg.DETECTOR_EXPLAINS.items():
            hit = df["anomaly_type"].isin(labels)
            pred = set(flags.loc[flags["detector"] == det, "txn_id"]) if len(flags) else set()
            if det == "class_mismatch":
                truth, pred, unit = set(df.loc[hit, "tag_id"]), set(tag_of.reindex(list(pred)).dropna()), "tag"
            else:
                truth, unit = set(df.loc[hit, "txn_id"]), "transaction"
            tp = len(truth & pred)
            out[det] = {"flagged": len(pred), "labelled": len(truth), "true_positives": tp,
                        "precision": round(tp / len(pred), 4) if pred else None,
                        "recall": round(tp / len(truth), 4) if truth else None, "unit": unit}
        return out
