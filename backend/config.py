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

# Severity cut-offs on the 0-100 risk score (policy choice, not a statistic)
CRITICAL_SCORE = 80
WARNING_SCORE = 50


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

# Which ground-truth labels each detector is responsible for explaining.
# The fare model looks at billed-vs-expected fare, so it legitimately also
# catches class_mismatch (a truck billed as a car is under-billed for its axles).
DETECTOR_EXPLAINS = {
    "cloned_tag": {"cloned_tag"},
    "class_mismatch": {"class_mismatch"},
    "duplicate_charge": {"duplicate_charge"},
    "fare_anomaly": {"fare_tampering", "class_mismatch"},
    "api_flood": {"api_flood"},
}



# ----------------------------------------------------------------------------
# Model features (single source of truth: used by train.py AND the live detector,
# so training and serving can never drift apart)
# ----------------------------------------------------------------------------
def flood_features(df):
    """API-flood model input: log-scaled request rate and CPU usage."""
    return pd.DataFrame({
        "log_req": np.log1p(df["req_per_min"].to_numpy(dtype=float)),
        "log_cpu": np.log1p(df["cpu_usage_percent"].to_numpy(dtype=float)),
    }, index=df.index)


def fare_features(df, plazas):
    """Fare model input: billed / expected fare, for the declared class and for
    the class implied by the axle count. A fare problem moves both ratios together;
    a class mismatch moves only the axle-based one."""
    factor = df["plaza_id"].map(plazas.set_index("plaza_id")["fare_factor"]).to_numpy(dtype=float)
    amount = df["amount_INR"].to_numpy(dtype=float)
    exp_declared = expected_fare(df["declared_class"], factor)
    exp_axle = expected_fare(axle_to_class(df["axle_count"]), factor)
    return pd.DataFrame({
        "fare_ratio": amount / exp_declared,
        "axle_fare_ratio": amount / exp_axle,
    }, index=df.index)



def above_baseline(pipe, feats):
    """Flood gate: only a request rate ABOVE the training average can be a flood.
    Isolation Forest also flags unusually quiet gateways, which are not attacks.
    The baseline is the mean the model's scaler learned from the training data."""
    return feats["log_req"].to_numpy() > pipe.named_steps["scaler"].mean_[0]