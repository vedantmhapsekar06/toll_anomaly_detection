# TollGuard

## Overview

**Problem statement.** Electronic toll collection (FASTag) on Indian highways is exposed to fraud and abuse: cloned tags, vehicles billed at a lower class than they are, duplicate deductions, wrong fares, and request floods against plaza systems. Toll operators log millions of transactions and cannot review them by hand. TollGuard ingests toll-plaza transaction logs, detects these anomalies, gives each one a risk score and a plain-English reason, and lets an analyst confirm, dismiss or block tags from a web dashboard.

**Motivation.** I built this to apply anomaly detection to infrastructure security, the area of my cybersecurity honours track. It also let me practise engineering choices that matter in production: rules where the definition is exact, calibrated ML where it is not, a held-out evaluation, and a deployable API with a usable interface.

**Data.** Real toll logs are not public, so the dataset is synthetic: about 50,000 transactions over 30 days, 3,388 tags and 19 plazas on Maharashtra highway corridors (plaza coordinates are approximate and illustrative). Vehicles follow realistic routes, so crossing times equal distance divided by speed, with hourly and weekday traffic patterns. 13% of transactions are injected, labelled anomalies, plus deliberate legitimate load spikes that cause honest false positives. A QA step re-derives the rules from the finished data and asserts they match the injected labels. The `anomaly_type` label is used only for evaluation, never for detection.

## What it detects

| Detector | Type | Signal |
|---|---|---|
| `cloned_tag` | Rule | The same tag at two plazas faster than any vehicle could travel (haversine distance / time > 140 km/h) |
| `class_mismatch` | Rule | The tag's declared class differs from the class implied by the axle count (a truck on a car tag) |
| `duplicate_charge` | Rule | The same tag charged again at the same plaza within 60 seconds |
| `fare_anomaly` | Isolation Forest | Billed amount is abnormal relative to the expected fare, for both the declared class and the axle-derived class |
| `api_flood` | Isolation Forest | Gateway request rate and CPU far above normal load |

## Dashboard
<img width="1097" height="630" alt="image" src="https://github.com/user-attachments/assets/eb5001eb-cd9a-415a-8b9a-bb5b0c35aa1c" />


## How scoring works

Each flag carries a probability and a plain-English reason.

- **Rules:** the probability is their measured, Laplace-smoothed precision.
- **ML detectors:** raw Isolation Forest scores are calibrated with isotonic regression into `P(anomaly | score)`, measured on a held-out calibration window. A row is flagged when that probability is at least 0.5.
- **Risk score (0-100)** = probability x impact weight, combined across detectors with noisy-OR. Impact weights are a documented policy choice: cloned tag 1.0, API flood 0.9, class mismatch 0.8, fare anomaly 0.7, duplicate charge 0.6.
- **Confidence** is the combined probability alone.
- **Severity:** Critical (80 and above), Warning (50-79), Info (below 50). These cut-offs are a policy choice, not a statistic.

## Results

Machine-learning detectors on the held-out test window (the last 25% of time, never seen during fitting or calibration):

| Detector | Precision | Recall | F1 | Precision at default IF threshold |
|---|---|---|---|---|
| `api_flood` | 97.2% | 100.0% | 98.6% | 13.5% |
| `fare_anomaly` | 100.0% | 98.7% | 99.4% | 64.9% |

Calibration is what turns Isolation Forest from a high-false-alarm default into a usable detector. The rule detectors reach 100% precision and recall on the full synthetic dataset by construction, so treat that as a consistency check, not a real-world claim. The pipeline scores 50,000 transactions in about 2 seconds.

## Architecture

```
CSV upload -> FastAPI -> column standardiser + validator
                      -> 3 rule detectors + 2 calibrated Isolation Forests
                      -> scoring + merge (one result per transaction)
                      -> storage (in-memory by default, optional MongoDB persistence)
                      -> dashboard (HTML / CSS / JS)
```

## Quick start

```bash
git clone https://github.com/vedantmhapsekar06/tollguard.git
cd tollguard
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r backend/requirements.txt

python ml/generate_data.py       # synthetic data + QA check
python ml/train.py               # trains and calibrates the models, writes metrics.json

uvicorn backend.main:app --reload
```

Open http://127.0.0.1:8000 and click **Run demo data**, or upload your own CSV (`ml/data/toll_demo_upload.csv` is a ready sample). Interactive API docs are at http://127.0.0.1:8000/docs. The repository already contains the generated data and trained models, so you can skip the two `ml/` commands.

**Storage:** by default alerts are kept in memory and reset when the server restarts, which is all you need to try the project. For persistence across restarts, start a MongoDB server and the app uses it automatically (`MONGO_URI`, default `mongodb://localhost:27017`). The dashboard header shows which backend is active.

The notebooks in `ml/notebooks/` need `pip install matplotlib seaborn ipykernel`.

## Input format

Required columns: `txn_id, tag_id, plaza_id, timestamp, declared_class, axle_count, amount_INR`.

Optional columns: `req_per_min, cpu_usage_percent` (enable the API-flood check), `lane_id, ip_address`, and `anomaly_type` (ground truth, enables a live accuracy check). Common column-name variants are mapped automatically, and a missing required column returns a clear error.

## Project structure

```
tollguard/
├── backend/
│   ├── main.py              # FastAPI routes, storage layer (memory / MongoDB)
│   ├── anomaly_detector.py  # detectors, scoring, merging
│   ├── config.py            # thresholds, fare table, features, paths
│   └── requirements.txt
├── ml/
│   ├── generate_data.py     # synthetic data + anomaly injection + QA
│   ├── train.py             # trains and calibrates models, writes metrics
│   ├── data/                # transactions, plazas, demo upload
│   ├── models/              # trained models, calibrators, metrics.json
│   └── notebooks/           # EDA and model analysis
├── frontend/                # index.html, style.css, app.js
└── assets/dashboard.png
```

## Design decisions

- **Rules vs ML:** exact definitions (impossible travel, wrong axle class, duplicate window) are rules. A threshold-sensitivity analysis shows the 140 km/h limit sits inside a wide plateau (100-200 km/h) of perfect precision and recall, so it is not a fragile choice.
- **Chronological split:** the models never see the future. Labels are used only to calibrate and evaluate.
- **Directional flood gate:** Isolation Forest also flags unusually quiet gateways, but a flood can only be high load. The flood detector flags only above-average load, which removed 71 false positives.
- **Probability x impact:** with probability alone, 97% of alerts came out Critical, which is useless for triage. Separating "is it real" from "how bad is it" fixed that.
- **One source of truth for features:** training and serving share the same feature functions in `config.py`, so they cannot drift apart.

## Limitations

- The data is synthetic. Metrics show the detectors find what was injected, not real-world performance. The method transfers to real labelled data, and the numbers would be re-fitted.
- Impact weights and severity cut-offs are policy choices, not learned.
- The fare model's weak spot is mild overcharges (1.4-1.6x the fare), which sit close to normal billing.
- Plaza coordinates are approximate.
- MongoDB persistence is optional, and that code path was verified against a mocked MongoDB, not a live server.



 ## Vedant Mhapsekar
