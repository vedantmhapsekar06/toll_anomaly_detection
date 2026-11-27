"""Synthetic FASTag toll transaction generator for TollGuard.

Run from the project root:  python ml/generate_data.py

Design
------
1. Normal traffic: ~3,400 vehicles (tags), each with a fixed class and home
   corridor, make trips along real Maharashtra highway corridors. Plaza
   crossing times follow distance / speed, so every normal tag moves
   physically plausibly. Traffic follows hourly and weekday profiles.
2. Legitimate messiness that detectors must NOT flag: random 50% concessions,
   local-pass tags that are consistently discounted, a genuine tariff revision at
   three plazas, axle-sensor misreads (1% of passages), per-plaza clock skew,
   isolated telemetry blips, short benign load surges and longer busy periods.
3. Anomalies (~13% of rows) are injected with variation, never one fixed pattern.
   Clones can land anywhere in a tag's history (before OR after genuine passages),
   floods overlap benign spikes in amplitude and only differ in how long they last,
   and fare tampering includes mild deviations.
4. Labels come from the injection process, NOT from the detection rules. The QA step
   therefore checks that the generator is internally valid (injected rows satisfy their
   definition, normal traffic stays physically plausible); it does not assert that a
   rule reproduces the labels.

Plaza names / coordinates are approximate and illustrative.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import config as cfg  # noqa: E402

rng = np.random.default_rng(cfg.SEED)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
START = pd.Timestamp("2026-10-01")
N_DAYS = 30
END = START + pd.Timedelta(days=N_DAYS)
START_SEC = int(START.timestamp())
END_SEC = int(END.timestamp())

TAGS_PER_CLASS = {"car": 2600, "lcv": 300, "truck": 350, "multi_axle": 150}
TRIPS_MEAN = {"car": 5.5, "lcv": 9.5, "truck": 14, "multi_axle": 14}   # trips per tag per 30 days
SPEED_MEAN = {"car": 80, "lcv": 65, "truck": 55, "multi_axle": 50}     # km/h
ROAD_FACTOR = 1.25          # road distance ~ 1.25 x straight-line distance
MIN_TRIP_GAP_S = 3600       # a tag starts a new trip >= 1 h after the last one ended
REPOSITION_KMH = 50         # conservative speed used to budget travel between trips
DISCOUNT_RATE = 0.08        # share of normal rows billed at 50% (return-journey concession)
BENIGN_SPIKE_RATE = 0.003   # isolated legit telemetry blips (single rows, not attacks)
N_SURGES = 40               # short benign load surges: 2-3 consecutive rows at one plaza
N_RUSHES = 12               # longer benign busy periods: 15-30 rows at 2-3.5x normal load
AXLE_ERROR_RATE = 0.01      # lane sensor misreads the axle count on 1% of passages
PASS_TAG_SHARE = 0.05       # share of car/lcv tags on a local pass ...
PASS_RATIO = 0.6            # ... billed at 60% on every crossing (legit, consistent per tag)
REVISION_PLAZAS = ["P03", "P11", "P16"]   # genuine tariff revision (+10%) at these plazas ...
REVISION_DAY = 15                         # ... from this day of the month onwards
REVISION_FACTOR = 1.10
PLAZA_SKEW_S = 30           # per-plaza clock offset (std, seconds)

N_MISMATCH_TAGS = 55        # tags that carry a lower-class tag than their axles
N_CLONE = 1000
N_DUP = 1500
N_FARE = 1400
N_BURSTS = 160              # api_flood windows (5-12 transactions each)

# ----------------------------------------------------------------------------
# Plazas and corridors  (id, name, corridor, lat, lon, fare_factor)
# ----------------------------------------------------------------------------
PLAZAS = [
    ("P01", "Vashi", "Mumbai-Kolhapur", 19.0760, 72.9990, 1.00),
    ("P02", "Khalapur", "Mumbai-Kolhapur", 18.8200, 73.2900, 1.25),
    ("P03", "Urse", "Mumbai-Kolhapur", 18.7150, 73.6000, 1.20),
    ("P04", "Khed Shivapur", "Mumbai-Kolhapur", 18.3300, 73.8400, 1.05),
    ("P05", "Anewadi", "Mumbai-Kolhapur", 17.9500, 73.9900, 1.00),
    ("P06", "Tasawade", "Mumbai-Kolhapur", 17.2600, 74.1900, 0.95),
    ("P07", "Kini", "Mumbai-Kolhapur", 16.8600, 74.3000, 0.90),
    ("P08", "Airoli", "Mumbai-Nashik", 19.1500, 72.9980, 0.90),
    ("P09", "Padgha", "Mumbai-Nashik", 19.3500, 73.1300, 0.95),
    ("P10", "Shahapur", "Mumbai-Nashik", 19.4500, 73.3300, 1.00),
    ("P11", "Ghoti", "Mumbai-Nashik", 19.7200, 73.6700, 1.05),
    ("P12", "Pimpalgaon Baswant", "Mumbai-Nashik", 20.1500, 73.9800, 1.10),
    ("P13", "Dahisar", "Mumbai-Gujarat", 19.2600, 72.8700, 0.85),
    ("P14", "Charoti", "Mumbai-Gujarat", 19.8700, 72.9300, 1.15),
    ("P15", "Yavat", "Pune-Solapur", 18.4600, 74.3900, 0.95),
    ("P16", "Surdi", "Pune-Solapur", 17.9000, 75.7000, 1.00),
    ("P17", "Khed", "Pune-Nashik", 18.8600, 73.8800, 0.95),
    ("P18", "Hivargaon Pawsa", "Pune-Nashik", 19.5800, 74.2200, 1.00),
    ("P19", "Gonde", "Pune-Nashik", 19.8900, 73.9500, 0.90),
]

# route -> (ordered plaza ids, share of tags)
ROUTES = {
    "MUM-KOP": (["P01", "P02", "P03", "P04", "P05", "P06", "P07"], 0.38),
    "MUM-NSK": (["P08", "P09", "P10", "P11", "P12"], 0.18),
    "MUM-GUJ": (["P13", "P14"], 0.14),
    "PUN-SOL": (["P15", "P16"], 0.10),
    "PUN-NSK": (["P17", "P18", "P19"], 0.20),
}

# hourly traffic shape (index = hour of day)
HOUR_CAR = np.array([1, .7, .5, .5, .8, 1.8, 3.2, 4.6, 5.0, 4.4, 3.6, 3.2,
                     3.4, 3.2, 3.2, 3.6, 4.2, 5.0, 5.2, 4.4, 3.2, 2.4, 1.8, 1.3])
HOUR_HEAVY = np.array([3.2, 3.0, 2.8, 2.8, 2.8, 2.6, 2.6, 2.6, 2.8, 3.0, 3.2, 3.4,
                       3.4, 3.2, 3.2, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.2, 4.0, 3.6])
# weekday shape (Mon..Sun): cars peak Fri-Sun, heavy vehicles dip on Sunday
WEEK_CAR = np.array([.9, .9, .9, 1.0, 1.3, 1.4, 1.3])
WEEK_HEAVY = np.array([1.0, 1.0, 1.0, 1.0, 1.0, .9, .7])


def _norm(x):
    x = np.asarray(x, dtype=float)
    return x / x.sum()


days_wd = np.array([(START + pd.Timedelta(days=i)).weekday() for i in range(N_DAYS)])
HOUR_P = {False: _norm(HOUR_CAR), True: _norm(HOUR_HEAVY)}
DAY_P = {False: _norm(WEEK_CAR[days_wd]), True: _norm(WEEK_HEAVY[days_wd])}
HOUR_LOAD = np.clip(HOUR_CAR / HOUR_CAR.mean(), 0.4, 1.6)   # gateway load by hour


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def build_plazas() -> pd.DataFrame:
    return pd.DataFrame(PLAZAS, columns=["plaza_id", "name", "corridor", "lat", "lon", "fare_factor"])


def build_tags() -> pd.DataFrame:
    n_total = sum(TAGS_PER_CLASS.values())
    ids = rng.choice(np.arange(100000, 999999), size=n_total, replace=False)
    route_ids = list(ROUTES)
    route_p = _norm([ROUTES[r][1] for r in route_ids])
    rows, k = [], 0
    for cls, count in TAGS_PER_CLASS.items():
        for _ in range(count):
            axles = {"car": 2, "lcv": 3, "truck": 4}.get(cls, int(rng.choice([5, 6], p=[0.6, 0.4])))
            rows.append({"tag_id": f"TAG{ids[k]}", "true_class": cls, "axle_count": axles,
                         "route": str(rng.choice(route_ids, p=route_p))})
            k += 1
    tags = pd.DataFrame(rows)
    tags["declared_class"] = tags["true_class"]
    # local-pass tags: legitimately and consistently discounted (cars and lcvs only)
    lite = tags["true_class"].isin(["car", "lcv"]).to_numpy()
    tags["pass_ratio"] = np.where(lite & (rng.random(len(tags)) < PASS_TAG_SHARE), PASS_RATIO, 1.0)

    # class-mismatch tags: a heavier vehicle running on a lower-class tag
    heavy = tags.index[tags["true_class"] != "car"].to_numpy()
    for i in rng.choice(heavy, size=N_MISMATCH_TAGS, replace=False):
        lower = cfg.CLASSES[: cfg.CLASS_RANK[tags.at[i, "true_class"]]]
        tags.at[i, "declared_class"] = str(rng.choice(lower))
    return tags


def draw_telemetry(hours):
    """Normal gateway telemetry: request rate follows traffic load, CPU follows rate."""
    hours = np.asarray(hours)
    req = np.round(rng.lognormal(np.log(55 * HOUR_LOAD[hours]), 0.30))
    cpu = np.clip(10 + 0.2 * req + rng.normal(0, 4, len(hours)), 5, 75).round(2)
    return req, cpu


def consecutive_window(plaza_arr, avail, k):
    """k consecutive (in time) available rows at one plaza, plaza picked by traffic share.
    Returns (plaza_id, row positions) or (None, None) if no plaza has enough rows."""
    counts = pd.Series(plaza_arr[avail]).value_counts()
    pid = str(rng.choice(counts.index, p=_norm(counts.to_numpy())))
    arr = np.flatnonzero(avail & (plaza_arr == pid))
    if len(arr) <= k:
        return None, None
    pos = int(rng.integers(0, len(arr) - k))
    return pid, arr[pos:pos + k]


def gateway_ip(plaza_ids, lanes):
    nums = pd.Series(plaza_ids).str[1:].astype(int).to_numpy()
    return np.array([f"10.20.{n}.{l}" for n, l in zip(nums, lanes)])


# ----------------------------------------------------------------------------
# Normal traffic
# ----------------------------------------------------------------------------
def simulate_trips(tags: pd.DataFrame, plazas: pd.DataFrame) -> pd.DataFrame:
    coord = plazas.set_index("plaza_id")[["lat", "lon"]]
    road_km = {}
    for route_ids, _ in ROUTES.values():
        for a in route_ids:
            for b in route_ids:
                if a != b:
                    road_km[(a, b)] = float(cfg.haversine_km(
                        coord.at[a, "lat"], coord.at[a, "lon"],
                        coord.at[b, "lat"], coord.at[b, "lon"])) * ROAD_FACTOR

    rows = []
    for t in tags.itertuples(index=False):
        n = rng.poisson(TRIPS_MEAN[t.true_class])
        if n == 0:
            continue
        heavy = t.true_class != "car"
        days = rng.choice(N_DAYS, n, p=DAY_P[heavy])
        hrs = rng.choice(24, n, p=HOUR_P[heavy])
        secs = rng.integers(0, 3600, n)
        starts = np.sort(START_SEC + days * 86400 + hrs * 3600 + secs)

        route = ROUTES[t.route][0]
        n_pl = len(route)
        len_p = _norm(1.0 / np.arange(1, n_pl + 1))     # shorter trips are more common
        prev_end, prev_pid = -np.inf, None
        for s in starts:
            L = int(rng.choice(np.arange(1, n_pl + 1), p=len_p))
            i0 = int(rng.integers(0, n_pl - L + 1))
            seg = route[i0:i0 + L]
            if rng.random() < 0.5:
                seg = seg[::-1]
            # next trip must start after the last one ended PLUS the time to
            # drive from where it ended to where it starts (no teleporting)
            reposition = 0.0
            if prev_pid is not None and prev_pid != seg[0]:
                reposition = road_km[(prev_pid, seg[0])] / REPOSITION_KMH * 3600
            if s < prev_end + MIN_TRIP_GAP_S + reposition:
                continue
            speed = float(np.clip(rng.normal(SPEED_MEAN[t.true_class], 8), 35, 105))
            ts = float(s)
            for k, pid in enumerate(seg):
                if k > 0:
                    ts += road_km[(seg[k - 1], pid)] / speed * 3600 + rng.normal(0, 20)
                if ts >= END_SEC:
                    break
                rows.append((t.tag_id, pid, ts))
                prev_pid = pid
            prev_end = ts
    return pd.DataFrame(rows, columns=["tag_id", "plaza_id", "ts_sec"])


def assemble_normal(trips: pd.DataFrame, tags: pd.DataFrame, plazas: pd.DataFrame, skew: pd.Series) -> pd.DataFrame:
    t = tags.set_index("tag_id")
    df = trips.copy()
    ts = df["ts_sec"] + df["plaza_id"].map(skew)              # each plaza's clock is slightly off
    df["timestamp"] = pd.to_datetime(ts.round().astype("int64"), unit="s")
    df = df.drop(columns="ts_sec")
    df["declared_class"] = df["tag_id"].map(t["declared_class"])
    df["axle_count"] = df["tag_id"].map(t["axle_count"]).astype(int)
    df = df.sort_values(["timestamp", "tag_id"], kind="stable").reset_index(drop=True)
    n = len(df)

    # lane sensors occasionally misread the axle count (+-1); the tag is unchanged
    err = rng.random(n) < AXLE_ERROR_RATE
    df.loc[err, "axle_count"] = np.maximum(2, df.loc[err, "axle_count"] + rng.choice([-1, 1], err.sum()))

    # legitimate billing: tariff x (random 50% concession | local-pass discount) x tariff revision
    factor = df["plaza_id"].map(plazas.set_index("plaza_id")["fare_factor"])
    ratio = np.where(rng.random(n) < DISCOUNT_RATE, 0.5, 1.0)
    pass_ratio = df["tag_id"].map(t["pass_ratio"]).to_numpy(dtype=float)
    ratio = np.where(pass_ratio < 1.0, pass_ratio, ratio)
    revised = (df["plaza_id"].isin(REVISION_PLAZAS) &
               (df["timestamp"] >= START + pd.Timedelta(days=REVISION_DAY))).to_numpy()
    ratio = ratio * np.where(revised, REVISION_FACTOR, 1.0)
    df["amount_INR"] = cfg.round_fare(cfg.expected_fare(df["declared_class"], factor) * ratio)

    df["lane_id"] = rng.integers(1, 7, n)
    df["ip_address"] = gateway_ip(df["plaza_id"], df["lane_id"])
    req, cpu = draw_telemetry(df["timestamp"].dt.hour.to_numpy())
    plaza_arr = df["plaza_id"].to_numpy()
    avail = np.ones(n, dtype=bool)

    # benign: isolated single-row blips (amplitude deliberately overlaps real floods)
    spike = rng.random(n) < BENIGN_SPIKE_RATE
    req[spike] = np.round(rng.uniform(250, 1500, spike.sum()))
    cpu[spike] = rng.uniform(45, 92, spike.sum()).round(2)
    # benign: short load surges, 2-3 consecutive rows at one plaza
    for _ in range(N_SURGES):
        _, sel = consecutive_window(plaza_arr, avail, int(rng.integers(2, 4)))
        if sel is None:
            continue
        avail[sel] = False
        req[sel] = np.round(rng.uniform(300, 900, len(sel)))
        cpu[sel] = rng.uniform(55, 90, len(sel)).round(2)
    # benign: longer busy periods, moderately elevated load for 15-30 rows
    for _ in range(N_RUSHES):
        _, sel = consecutive_window(plaza_arr, avail, int(rng.integers(15, 31)))
        if sel is None:
            continue
        avail[sel] = False
        req[sel] = np.round(req[sel] * rng.uniform(2.0, 3.5, len(sel)))
        cpu[sel] = np.clip(10 + 0.2 * req[sel] + rng.normal(0, 4, len(sel)), 5, 90).round(2)
    df["req_per_min"] = req
    df["cpu_usage_percent"] = cpu

    # tags whose declared class differs from their real class are class_mismatch
    mism = df["tag_id"].map(t["true_class"]) != df["declared_class"]
    df["anomaly_type"] = np.where(mism, "class_mismatch", "none")
    return df


# ----------------------------------------------------------------------------
# Anomaly injection
# ----------------------------------------------------------------------------
def inject(df: pd.DataFrame, plazas: pd.DataFrame, skew: pd.Series) -> pd.DataFrame:
    p = plazas.set_index("plaza_id")
    fare_factor = df["plaza_id"].map(p["fare_factor"]).to_numpy()
    expected = cfg.expected_fare(df["declared_class"], fare_factor)
    free = (df["anomaly_type"] == "none").to_numpy(copy=True)  # writable copy (pandas CoW)

    # --- fare_tampering: billed amount deviates from the tariff, mildly or badly --------
    idx = rng.choice(np.flatnonzero(free), N_FARE, replace=False)
    free[idx] = False
    kind = rng.choice(5, N_FARE, p=[0.20, 0.25, 0.15, 0.25, 0.15])
    ratio = np.select(
        [kind == 0, kind == 1, kind == 2, kind == 3],
        [np.zeros(N_FARE), rng.uniform(0.05, 0.35, N_FARE),      # free pass, heavy under-charge
         rng.uniform(0.65, 0.90, N_FARE),                        # mild under-charge
         rng.uniform(1.10, 1.50, N_FARE)],                       # mild over-charge
        default=rng.uniform(1.5, 3.0, N_FARE))                   # heavy over-charge
    df.loc[idx, "amount_INR"] = cfg.round_fare(expected[idx] * ratio)
    df.loc[idx, "anomaly_type"] = "fare_tampering"

    # --- api_flood: SUSTAINED telemetry surge over 5-12 consecutive rows at one plaza ----
    plaza_arr = df["plaza_id"].to_numpy()
    for _ in range(N_BURSTS):
        _, sel = consecutive_window(plaza_arr, free, int(rng.integers(5, 13)))
        if sel is None:
            continue
        free[sel] = False
        df.loc[sel, "req_per_min"] = np.round(np.exp(rng.uniform(np.log(400), np.log(5000), len(sel))))
        df.loc[sel, "cpu_usage_percent"] = rng.uniform(60, 99, len(sel)).round(2)
        attacker = ".".join(str(x) for x in (rng.choice([45, 103, 139, 185]), *rng.integers(1, 255, 3)))
        df.loc[sel, "ip_address"] = attacker
        df.loc[sel, "anomaly_type"] = "api_flood"

    # --- duplicate_charge: same tag + plaza + amount a few seconds later ------
    dup_idx = rng.choice(np.flatnonzero(free), N_DUP, replace=False)
    free[dup_idx] = False
    dups = df.iloc[dup_idx].copy()
    dups["timestamp"] = dups["timestamp"] + pd.to_timedelta(rng.integers(3, 56, N_DUP), unit="s").to_numpy()
    dups["anomaly_type"] = "duplicate_charge"

    # --- cloned_tag: a genuine passage somewhere in the tag's history, plus an impossible
    #     passage far away, either BEFORE or AFTER it, so genuine passages can follow a clone
    coord = p[["lat", "lon"]]
    pool = rng.permutation(np.flatnonzero(free))
    clones = []
    for i in pool:
        if len(clones) >= N_CLONE:
            break
        base = df.iloc[i]
        a = base["plaza_id"]
        dist = cfg.haversine_km(coord.at[a, "lat"], coord.at[a, "lon"], coord["lat"], coord["lon"])
        far = dist.index[dist.to_numpy() >= 100]
        b = str(rng.choice(far))
        speed = rng.uniform(200, 900)                         # impossible implied speed, km/h
        delta = float(dist[b]) / speed * 3600 + skew[b] - skew[a]
        sign = 1 if rng.random() < 0.5 else -1
        ts = base["timestamp"] + pd.Timedelta(seconds=round(sign * delta))
        if not (START <= ts < END):
            continue
        row = base.copy()
        row["plaza_id"] = b
        row["timestamp"] = ts
        row["lane_id"] = int(rng.integers(1, 7))
        row["ip_address"] = gateway_ip([b], [row["lane_id"]])[0]
        row["amount_INR"] = float(cfg.expected_fare([row["declared_class"]], [p.at[b, "fare_factor"]])[0])
        row["anomaly_type"] = "cloned_tag"
        clones.append(row)
        free[i] = False
    clones = pd.DataFrame(clones)
    req, cpu = draw_telemetry(clones["timestamp"].dt.hour.to_numpy())
    clones["req_per_min"], clones["cpu_usage_percent"] = req, cpu

    out = pd.concat([df, dups, clones], ignore_index=True)
    out = out.sort_values(["timestamp", "tag_id"], kind="stable").reset_index(drop=True)
    out.insert(0, "txn_id", [f"TXN{i:07d}" for i in range(1, len(out) + 1)])
    return out


# ----------------------------------------------------------------------------
# QA: re-derive detection rules from the data and compare with injected labels
# ----------------------------------------------------------------------------
def qa_checks(df: pd.DataFrame, plazas: pd.DataFrame) -> None:
    """Generator validity. Labels are NOT re-derived from the detection rules (that would be
    circular); these checks only confirm the data is what the generator claims it is."""
    # 1. normal traffic must be physically plausible, or "impossible travel" means nothing
    d = df[df["anomaly_type"] != "cloned_tag"].merge(plazas[["plaza_id", "lat", "lon"]], on="plaza_id")
    d = d.sort_values(["tag_id", "timestamp"], kind="stable")
    g = d.groupby("tag_id")
    dist = cfg.haversine_km(g["lat"].shift(), g["lon"].shift(), d["lat"], d["lon"])
    hours = (d["timestamp"] - g["timestamp"].shift()).dt.total_seconds() / 3600
    top = float((dist / hours.clip(lower=1 / 3600)).max())
    assert top < cfg.MAX_SPEED_KMH, f"normal traffic reaches {top:.0f} km/h"

    # 2. injected rows satisfy their own definition
    dup = df[df["anomaly_type"] == "duplicate_charge"]
    assert len(dup) == N_DUP and dup["amount_INR"].gt(-1).all()
    assert (df.loc[df["anomaly_type"] == "api_flood", "req_per_min"] >= 400).all()
    assert (df["anomaly_type"] == "fare_tampering").sum() == N_FARE
    assert (df["anomaly_type"] == "cloned_tag").sum() == N_CLONE

    # 3. bookkeeping
    assert df["txn_id"].is_unique
    assert df["timestamp"].min() >= START - pd.Timedelta(minutes=10)
    assert df["timestamp"].max() < END + pd.Timedelta(days=1)
    print(f"QA passed: normal traffic tops out at {top:.0f} km/h (limit {cfg.MAX_SPEED_KMH}), "
          f"injected counts and ranges as designed")


# ----------------------------------------------------------------------------
# Demo file: the last DEMO_DAYS of FULL traffic. A real upload is a time slice of all
# plazas, not a sample of tags, and the flood and fare detectors read each plaza's
# consecutive traffic. This window falls inside train.py's held-out test period, so the
# "live check" on the demo is a genuine out-of-sample result.
# ----------------------------------------------------------------------------
DEMO_DAYS = 4


def make_demo(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["timestamp"] >= END - pd.Timedelta(days=DEMO_DAYS)].reset_index(drop=True)


# ----------------------------------------------------------------------------
COLS = ["txn_id", "tag_id", "plaza_id", "lane_id", "timestamp", "declared_class", "axle_count",
        "amount_INR", "ip_address", "req_per_min", "cpu_usage_percent", "anomaly_type"]


def generate(seed: int = cfg.SEED, verbose: bool = True):
    """Build one complete synthetic world and return (transactions, plazas).
    Pure function of `seed`: nothing is written, so the pipeline can create several
    independent worlds (new vehicles, trips and anomalies) to test the models on data they
    have never seen."""
    global rng
    rng = np.random.default_rng(seed)
    plazas = build_plazas()
    tags = build_tags()
    skew = pd.Series(rng.normal(0, PLAZA_SKEW_S, len(plazas)), index=plazas["plaza_id"])
    trips = simulate_trips(tags, plazas)
    df = assemble_normal(trips, tags, plazas, skew)
    df = inject(df, plazas, skew)
    df = df[COLS].copy()
    df["amount_INR"] = df["amount_INR"].round().astype(int)
    df["req_per_min"] = df["req_per_min"].round().astype(int)
    df["lane_id"] = df["lane_id"].astype(int)
    if verbose:
        qa_checks(df, plazas)
    else:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            qa_checks(df, plazas)
    return df, plazas


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Generate the synthetic TollGuard dataset")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--out-dir", type=Path, default=cfg.DATA_DIR,
                    help="where to write the CSVs (default: ml/data)")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    df, plazas = generate(args.seed)
    df.to_csv(out / cfg.TXN_CSV.name, index=False, date_format="%Y-%m-%d %H:%M:%S")
    plazas.to_csv(out / cfg.PLAZA_CSV.name, index=False)
    demo = make_demo(df)
    demo.to_csv(out / cfg.DEMO_CSV.name, index=False, date_format="%Y-%m-%d %H:%M:%S")

    print(f"\nSaved {len(df):,} rows -> {out / cfg.TXN_CSV.name}")
    print(f"Saved {len(demo):,} rows -> {out / cfg.DEMO_CSV.name}")
    print(f"Unique tags: {df['tag_id'].nunique():,} | plazas: {df['plaza_id'].nunique()} | "
          f"{df['timestamp'].min()} to {df['timestamp'].max()}")
    vc = df["anomaly_type"].value_counts()
    print("\nLabel distribution:")
    for k, v in vc.items():
        print(f"  {k:<18}{v:>7,}  {v / len(df):6.2%}")
    print(f"  anomaly rate: {(df['anomaly_type'] != 'none').mean():.2%}")


if __name__ == "__main__":
    main()
