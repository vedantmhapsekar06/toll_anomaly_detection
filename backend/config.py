"""TollGuard shared configuration: paths, thresholds, fare table, helpers.

Every threshold used by the generator, trainer and detectors lives here so that
each number can be justified in one place.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ml" / "data"
MODEL_DIR = ROOT / "ml" / "models"
FRONTEND_DIR = ROOT / "frontend"

TXN_CSV = DATA_DIR / "toll_transactions.csv"
DEMO_CSV = DATA_DIR / "toll_demo_upload.csv"
PLAZA_CSV = DATA_DIR / "plazas.csv"

FLOOD_MODEL = MODEL_DIR / "flood_model.pkl"
FARE_MODEL = MODEL_DIR / "fare_model.pkl"
CALIBRATION = MODEL_DIR / "calibration.pkl"
METRICS_JSON = MODEL_DIR / "metrics.json"
SELECTION_JSON = MODEL_DIR / "selection.json"
DIAGNOSTICS_JSON = MODEL_DIR / "diagnostics.json"

SEED = 42

# ----------------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------------
REQUIRED_COLUMNS = [
    "txn_id", "tag_id", "plaza_id", "timestamp",
    "declared_class", "axle_count", "amount_INR",
]
TELEMETRY_COLUMNS = ["req_per_min", "cpu_usage_percent"]   # needed for api_flood
OPTIONAL_COLUMNS = ["lane_id", "ip_address", "anomaly_type"]  # anomaly_type = ground truth

# ----------------------------------------------------------------------------
# Vehicle classes and fares (simplified, axle-based classification)
# ----------------------------------------------------------------------------
CLASSES = ["car", "lcv", "truck", "multi_axle"]
CLASS_RANK = {c: i for i, c in enumerate(CLASSES)}
BASE_FARE = {"car": 100, "lcv": 180, "truck": 350, "multi_axle": 600}  # INR
FARE_ROUND_TO = 5  # tolls are charged in multiples of Rs 5


def axle_to_class(axles):
    """Axle count -> vehicle class. 2 = car, 3 = lcv, 4 = truck, 5+ = multi_axle."""
    a = np.asarray(axles)
    return np.select([a <= 2, a == 3, a == 4], ["car", "lcv", "truck"], default="multi_axle")


def round_fare(x):
    return np.round(np.asarray(x, dtype=float) / FARE_ROUND_TO) * FARE_ROUND_TO


def expected_fare(classes, fare_factor):
    """Expected fare (INR) for vehicle class(es) at plaza(s) with the given fare factor."""
    base = pd.Series(classes).map(BASE_FARE).to_numpy(dtype=float)
    return round_fare(base * np.asarray(fare_factor, dtype=float))


# ----------------------------------------------------------------------------
# Geo
# ----------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars, numpy arrays and Series."""
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------------------
# Detection thresholds
# ----------------------------------------------------------------------------
# Expressway limit is 120 km/h; +~15% tolerance for timestamp noise = 140.
# Straight-line distance <= road distance, so implied speed is a LOWER BOUND of
# the true speed: anything above the limit is physically impossible.
MAX_SPEED_KMH = 140
DUPLICATE_WINDOW_S = 60     # same tag + same plaza within this many seconds

# Class mismatch is a property of a TAG, not of one passage: an axle sensor can misread a
# single passage, so a tag is flagged only when the mismatch is persistent.
CLASS_MISMATCH_MIN_PASSAGES = 2   # at least this many mismatching passages ...
CLASS_MISMATCH_MIN_SHARE = 0.5    # ... making up at least this share of the tag's passages

# Context windows (rows). A flood lasts several consecutive gateway rows while a benign
# blip is short, so the flood model sees the plaza's recent load, not just one row.
FLOOD_CTX_ROWS = 5
# Billing is judged against the tag's own history and the plaza's current tariff level.
FARE_CTX_ROWS = 41
FARE_TAG_MIN_HISTORY = 3
# Legitimate billing sits at a few discrete levels of the tariff (full fare, 50% return-journey
# concession). Isolation Forest scores how EXTREME a point is, so an amount that falls between
# two dense legitimate clusters (say 0.8x) looks ordinary to it. The gap to the nearest
# legitimate level makes "in between" explicit.
STANDARD_BILLING_LEVELS = (1.0, 0.5)

# Calibrated probabilities are only valid at the anomaly rate they were fitted on.
# To deploy where anomalies are rarer, set e.g. {"fare_anomaly": 0.005, "api_flood": 0.002};
# probabilities are then re-based with the standard prior-shift (odds) correction.
DEPLOYMENT_PREVALENCE: dict = {}

# Severity cut-offs on the 0-100 risk score (policy choice, not a statistic)
CRITICAL_SCORE = 80
WARNING_SCORE = 50


# Detectors are combined with noisy-OR, which assumes they carry independent evidence.
# That is why each detector looks at a different thing: identity (cloned_tag), declared vs
# physical class (class_mismatch), billed vs tariff (fare_anomaly), double deduction
# (duplicate_charge) and gateway load (api_flood). No two detectors read the same signal.
#
# Impact weight = how costly a CONFIRMED anomaly of this type is (policy choice).
# risk = 100 x P(anomaly is real) x impact, so a near-certain duplicate charge is
# less urgent than a near-certain cloned tag. Probabilities are measured; weights are policy.
IMPACT_WEIGHT = {
    "cloned_tag": 1.0,         # identity / security breach
    "api_flood": 0.9,          # availability attack on plaza systems
    "class_mismatch": 0.8,     # systematic revenue leakage
    "fare_anomaly": 0.7,       # wrong amount billed
    "duplicate_charge": 0.6,   # customer-facing billing error, reversible
}

# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------
ANOMALY_TYPES = ["cloned_tag", "class_mismatch", "duplicate_charge", "fare_tampering", "api_flood"]

# Which ground-truth labels each detector is responsible for explaining. One label per
# detector: the fare model judges the billed amount against the DECLARED class only, so a
# class mismatch (a truck on a car tag, billed correctly as a car) is not its business.
DETECTOR_EXPLAINS = {
    "cloned_tag": {"cloned_tag"},
    "class_mismatch": {"class_mismatch"},
    "duplicate_charge": {"duplicate_charge"},
    "fare_anomaly": {"fare_tampering"},
    "api_flood": {"api_flood"},
}



# ----------------------------------------------------------------------------
# Model features (single source of truth: used by train.py AND the live detector,
# so training and serving can never drift apart)
# ----------------------------------------------------------------------------
def _plaza_context(values: pd.Series, df: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    """Centered rolling median of `values` over each plaza's time-ordered rows.
    Returned on df's own index. Needs plaza_id and timestamp; df.index must be unique."""
    order = df.sort_values(["plaza_id", "timestamp"], kind="stable").index
    ctx = values.loc[order].groupby(df.loc[order, "plaza_id"], sort=False).transform(
        lambda s: s.rolling(window, center=True, min_periods=min_periods).median())
    return ctx.reindex(df.index)


def flood_features(df):
    """API-flood model input: the row's own load plus the plaza's recent load.
    A flood is SUSTAINED (several consecutive rows at one plaza); a benign blip is short.
    The context columns are what separates them when the amplitudes overlap.
    log_req must stay the first column (see above_baseline)."""
    log_req = pd.Series(np.log1p(df["req_per_min"].to_numpy(dtype=float)), index=df.index)
    log_cpu = pd.Series(np.log1p(df["cpu_usage_percent"].to_numpy(dtype=float)), index=df.index)
    return pd.DataFrame({
        "log_req": log_req,
        "log_cpu": log_cpu,
        "ctx_req": _plaza_context(log_req, df, FLOOD_CTX_ROWS, 3).fillna(log_req),
        "ctx_cpu": _plaza_context(log_cpu, df, FLOOD_CTX_ROWS, 3).fillna(log_cpu),
    }, index=df.index)


def _finite_or_one(x: pd.Series) -> pd.Series:
    return x.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def fare_features(df, plazas):
    """Fare model input, all relative to the tariff for the DECLARED class:
      fare_ratio : billed / tariff
      vs_tag     : fare_ratio relative to this tag's own usual ratio (a monthly-pass tag is
                   consistently discounted; a one-off deviation is what matters)
      vs_plaza   : fare_ratio relative to the plaza's recent level (a legitimate tariff
                   revision moves every vehicle at that plaza, not one)
      level_gap  : distance of the better of those two from the nearest legitimate billing
                   level (see STANDARD_BILLING_LEVELS)
    Declared-vs-axle class is deliberately NOT used here; class_mismatch has its own rule."""
    factor = df["plaza_id"].map(plazas.set_index("plaza_id")["fare_factor"]).to_numpy(dtype=float)
    expected = expected_fare(df["declared_class"], factor)
    ratio = pd.Series(df["amount_INR"].to_numpy(dtype=float) / expected, index=df.index)

    vs_plaza = _finite_or_one(ratio / _plaza_context(ratio, df, FARE_CTX_ROWS, 5))

    by_tag = ratio.groupby(df["tag_id"])
    tag_med = by_tag.transform("median")
    tag_n = by_tag.transform("size")
    # With too little history the tag says nothing, so fall back to the plaza reference.
    # (Falling back to 1.0 would read "no information" as "perfectly standard billing".)
    vs_tag = _finite_or_one(ratio / tag_med).where(tag_n >= FARE_TAG_MIN_HISTORY, vs_plaza)
    # keep NaN where the ratio itself is unknown (unknown plaza): those rows are skipped
    vs_tag, vs_plaza = vs_tag.where(ratio.notna()), vs_plaza.where(ratio.notna())
    levels = np.asarray(STANDARD_BILLING_LEVELS, dtype=float)

    def gap(x: pd.Series) -> np.ndarray:
        return np.abs(x.to_numpy(dtype=float)[:, None] - levels[None, :]).min(axis=1)
    # normal if standard relative to the tag's own history OR to the plaza's current level
    level_gap = pd.Series(np.minimum(gap(vs_tag), gap(vs_plaza)), index=df.index)
    return pd.DataFrame({"fare_ratio": ratio, "vs_tag": vs_tag, "vs_plaza": vs_plaza,
                         "level_gap": level_gap}, index=df.index)


def above_baseline(model, feats):
    """Flood gate: only a plaza load ABOVE the training average can be a flood.
    Anomaly detectors also flag unusually quiet gateways, which are not attacks.
    The baseline is the training mean of the plaza-load feature stored in the model."""
    return feats["ctx_req"].to_numpy() > model.gate_threshold


def shift_prior(p, train_prevalence: float, deploy_prevalence: float):
    """Re-base a calibrated probability from the anomaly rate it was fitted at to another one."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    odds = p / (1 - p) * (deploy_prevalence / (1 - deploy_prevalence)) / (train_prevalence / (1 - train_prevalence))
    return odds / (1 + odds)
