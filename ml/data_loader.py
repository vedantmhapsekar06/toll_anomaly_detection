"""Load and validate the toll transaction data. Every ML script reads data through here.

Run from the project root:  python ml/data_loader.py   (writes ml/data/data_report.json)

Hard checks raise an error (the data cannot be trusted); soft checks only land in the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import config as cfg  # noqa: E402

REQUIRED = ["txn_id", "tag_id", "plaza_id", "timestamp", "declared_class", "axle_count", "amount_INR"]
TELEMETRY = list(cfg.TELEMETRY_COLUMNS)
LABEL = "anomaly_type"
ALLOWED_LABELS = {"none", *cfg.ANOMALY_TYPES}


class DataValidationError(ValueError):
    pass


def validate(df: pd.DataFrame, plazas: pd.DataFrame) -> dict:
    """Return a report dict; raise DataValidationError on any hard failure."""
    errors = []
    missing = [c for c in REQUIRED + TELEMETRY + [LABEL] if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")

    nulls = df[REQUIRED + TELEMETRY + [LABEL]].isna().sum()
    if nulls.any():
        errors.append(f"null values: {nulls[nulls > 0].to_dict()}")
    if not df["txn_id"].is_unique:
        errors.append("txn_id is not unique")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        errors.append("timestamp is not datetime")
    if (df["amount_INR"] < 0).any():
        errors.append("negative amounts")
    if (df[TELEMETRY] < 0).any().any():
        errors.append("negative telemetry")
    unknown_plazas = set(df["plaza_id"]) - set(plazas["plaza_id"])
    if unknown_plazas:
        errors.append(f"unknown plazas: {sorted(unknown_plazas)[:5]}")
    unknown_labels = set(df[LABEL]) - ALLOWED_LABELS
    if unknown_labels:
        errors.append(f"unknown labels: {sorted(unknown_labels)}")
    bad_axles = ~df["axle_count"].between(2, 10)
    if bad_axles.any():
        errors.append(f"{int(bad_axles.sum())} rows with implausible axle counts")
    if errors:
        raise DataValidationError("; ".join(errors))

    counts = df[LABEL].value_counts()
    return {
        "rows": int(len(df)),
        "tags": int(df["tag_id"].nunique()),
        "plazas": int(df["plaza_id"].nunique()),
        "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "days": round(float((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400), 1),
        "label_counts": {k: int(v) for k, v in counts.items()},
        "anomaly_rate": round(float((df[LABEL] != "none").mean()), 4),
        "amount_INR": {"min": int(df["amount_INR"].min()), "median": float(df["amount_INR"].median()),
                       "max": int(df["amount_INR"].max())},
        "req_per_min": {"median": float(df["req_per_min"].median()), "p99": float(df["req_per_min"].quantile(.99)),
                        "max": int(df["req_per_min"].max())},
        "rows_per_tag": {"median": float(df.groupby("tag_id").size().median()),
                         "max": int(df.groupby("tag_id").size().max())},
        "checks_passed": ["required columns", "no nulls", "unique txn_id", "datetime timestamps",
                          "non-negative amounts and telemetry", "known plazas", "known labels",
                          "plausible axle counts"],
    }


def load_plazas(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or cfg.PLAZA_CSV)


def load_transactions(path: Path | None = None, plazas: pd.DataFrame | None = None,
                      check: bool = True) -> pd.DataFrame:
    """Time-ordered transactions with a clean 0..n-1 index (the model code relies on that)."""
    df = pd.read_csv(path or cfg.TXN_CSV, parse_dates=["timestamp"])
    if check:
        validate(df, plazas if plazas is not None else load_plazas())
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def main() -> None:
    plazas = load_plazas()
    df = load_transactions(plazas=plazas)
    report = validate(df, plazas)
    out = cfg.DATA_DIR / "data_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"validated {report['rows']:,} rows | {report['tags']:,} tags | {report['plazas']} plazas | "
          f"{report['days']} days | anomaly rate {report['anomaly_rate']:.1%}")
    print("checks passed:", ", ".join(report["checks_passed"]))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
