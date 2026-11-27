"""TollGuard API: upload a toll-transaction CSV, run the detector, store alerts,
let analysts confirm / dismiss / block, and serve the dashboard.

Run from the project root:   uvicorn backend.main:app --reload
Then open:                   http://127.0.0.1:8000

Storage: MongoDB if reachable (MONGO_URI, default mongodb://localhost:27017),
otherwise an in-memory store so the app still runs (data is lost on restart).
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument

from backend import config as cfg
from backend.anomaly_detector import TollGuardDetector

MAX_UPLOAD_MB = 25
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "tollguard")
Status = Literal["open", "confirmed", "dismissed", "blocked"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------
# Storage layer (same interface, two backends)
# ----------------------------------------------------------------------------
def _matches(a: dict, f: dict) -> bool:
    """In-memory equivalent of the Mongo filter built in MongoStore._filter."""
    if f.get("severity") and a["severity"] != f["severity"]:
        return False
    if f.get("detector") and f["detector"] not in a["detected_type"].split("+"):
        return False
    if f.get("plaza") and a["plaza_name"] != f["plaza"]:
        return False
    if f.get("status") and a["status"] != f["status"]:
        return False
    if f.get("q"):
        q = f["q"].lower()
        if q not in a["tag_id"].lower() and q not in a["txn_id"].lower():
            return False
    if f.get("min_score") is not None and a["risk_score"] < f["min_score"]:
        return False
    return True


class MemoryStore:
    name = "memory"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.runs: dict[str, dict] = {}
        self.alerts: dict[str, list[dict]] = {}
        self.blocked: dict[str, dict] = {}

    def save_run(self, run: dict, alerts: list[dict]) -> None:
        with self._lock:
            self.runs[run["run_id"]] = run
            self.alerts[run["run_id"]] = [dict(a) for a in alerts]

    def list_runs(self, limit: int = 20) -> list[dict]:
        return sorted(self.runs.values(), key=lambda r: r["created_at"], reverse=True)[:limit]

    def get_run(self, run_id: str) -> Optional[dict]:
        return self.runs.get(run_id)

    def query_alerts(self, run_id: str, f: dict, page: int, page_size: int) -> tuple[int, list[dict]]:
        rows = [a for a in self.alerts.get(run_id, []) if _matches(a, f)]
        rows.sort(key=lambda a: (-a["risk_score"], a["timestamp"]))
        start = (page - 1) * page_size
        return len(rows), rows[start:start + page_size]

    def status_counts(self, run_id: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.alerts.get(run_id, []):
            out[a["status"]] = out.get(a["status"], 0) + 1
        return out

    def update_alert(self, run_id: str, txn_id: str, status: str, note: str) -> Optional[dict]:
        with self._lock:
            for a in self.alerts.get(run_id, []):
                if a["txn_id"] == txn_id:
                    a.update(status=status, note=note, updated_at=now_iso())
                    return dict(a)
        return None

    def cascade_block(self, run_id: str, tag_id: str) -> int:
        n = 0
        with self._lock:
            for a in self.alerts.get(run_id, []):
                if a["tag_id"] == tag_id and a["status"] == "open":
                    a.update(status="blocked", updated_at=now_iso())
                    n += 1
        return n

    def block_tag(self, tag_id: str, run_id: str, txn_id: str) -> None:
        with self._lock:
            self.blocked[tag_id] = {"tag_id": tag_id, "run_id": run_id, "txn_id": txn_id,
                                    "blocked_at": now_iso()}

    def unblock_tag(self, tag_id: str) -> bool:
        with self._lock:
            return self.blocked.pop(tag_id, None) is not None

    def blocked_tags(self) -> list[dict]:
        return sorted(self.blocked.values(), key=lambda b: b["blocked_at"], reverse=True)

    def blocked_set(self) -> set[str]:
        return set(self.blocked)


class MongoStore:
    name = "mongodb"

    def __init__(self, client: MongoClient) -> None:
        db = client[MONGO_DB]
        self.runs, self.alerts, self.blocked = db["runs"], db["alerts"], db["blocked_tags"]
        self.alerts.create_index([("run_id", ASCENDING), ("txn_id", ASCENDING)], unique=True)
        self.alerts.create_index([("run_id", ASCENDING), ("risk_score", DESCENDING)])
        self.runs.create_index([("created_at", DESCENDING)])
        self.blocked.create_index("tag_id", unique=True)

    @staticmethod
    def _filter(run_id: str, f: dict) -> dict:
        q: dict = {"run_id": run_id}
        if f.get("severity"):
            q["severity"] = f["severity"]
        if f.get("detector"):
            q["detected_type"] = {"$regex": rf"(^|\+){re.escape(f['detector'])}($|\+)"}
        if f.get("plaza"):
            q["plaza_name"] = f["plaza"]
        if f.get("status"):
            q["status"] = f["status"]
        if f.get("q"):
            rx = {"$regex": re.escape(f["q"]), "$options": "i"}
            q["$or"] = [{"tag_id": rx}, {"txn_id": rx}]
        if f.get("min_score") is not None:
            q["risk_score"] = {"$gte": f["min_score"]}
        return q

    def save_run(self, run: dict, alerts: list[dict]) -> None:
        self.runs.insert_one(dict(run))
        docs = [dict(a) for a in alerts]          # insert_many mutates dicts (_id), so copy
        for i in range(0, len(docs), 5000):
            self.alerts.insert_many(docs[i:i + 5000], ordered=False)

    def list_runs(self, limit: int = 20) -> list[dict]:
        return list(self.runs.find({}, {"_id": 0}).sort("created_at", -1).limit(limit))

    def get_run(self, run_id: str) -> Optional[dict]:
        return self.runs.find_one({"run_id": run_id}, {"_id": 0})

    def query_alerts(self, run_id: str, f: dict, page: int, page_size: int) -> tuple[int, list[dict]]:
        flt = self._filter(run_id, f)
        total = self.alerts.count_documents(flt)
        cur = (self.alerts.find(flt, {"_id": 0}).sort([("risk_score", -1), ("timestamp", 1)])
               .skip((page - 1) * page_size).limit(page_size))
        return total, list(cur)

    def status_counts(self, run_id: str) -> dict[str, int]:
        rows = self.alerts.aggregate([{"$match": {"run_id": run_id}},
                                      {"$group": {"_id": "$status", "n": {"$sum": 1}}}])
        return {r["_id"]: r["n"] for r in rows}

    def update_alert(self, run_id: str, txn_id: str, status: str, note: str) -> Optional[dict]:
        return self.alerts.find_one_and_update(
            {"run_id": run_id, "txn_id": txn_id},
            {"$set": {"status": status, "note": note, "updated_at": now_iso()}},
            projection={"_id": 0}, return_document=ReturnDocument.AFTER)

    def cascade_block(self, run_id: str, tag_id: str) -> int:
        res = self.alerts.update_many({"run_id": run_id, "tag_id": tag_id, "status": "open"},
                                      {"$set": {"status": "blocked", "updated_at": now_iso()}})
        return res.modified_count

    def block_tag(self, tag_id: str, run_id: str, txn_id: str) -> None:
        self.blocked.update_one({"tag_id": tag_id},
                                {"$set": {"tag_id": tag_id, "run_id": run_id, "txn_id": txn_id,
                                          "blocked_at": now_iso()}}, upsert=True)

    def unblock_tag(self, tag_id: str) -> bool:
        return self.blocked.delete_one({"tag_id": tag_id}).deleted_count > 0

    def blocked_tags(self) -> list[dict]:
        return list(self.blocked.find({}, {"_id": 0}).sort("blocked_at", -1))

    def blocked_set(self) -> set[str]:
        return {b["tag_id"] for b in self.blocked.find({}, {"tag_id": 1, "_id": 0})}


def create_store():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        return MongoStore(client)
    except Exception as exc:  # noqa: BLE001 - any connection problem -> fall back
        print(f"[TollGuard] MongoDB not reachable at {MONGO_URI} ({type(exc).__name__}). "
              "Using IN-MEMORY storage: data is lost when the server restarts.")
        return MemoryStore()


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["detector"] = TollGuardDetector()
    except FileNotFoundError as exc:
        raise RuntimeError("Model files not found. Run:  python ml/generate_data.py  then  "
                           "python ml/train.py") from exc
    state.setdefault("store", create_store())
    print(f"[TollGuard] ready | storage: {state['store'].name}")
    yield


app = FastAPI(title="TollGuard", version="1.0", lifespan=lifespan)


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (NaN -> null, numpy -> python)."""
    return json.loads(df.to_json(orient="records"))


def _run_analysis(df_raw: pd.DataFrame, filename: str) -> dict:
    detector: TollGuardDetector = state["detector"]
    store = state["store"]
    try:
        out = detector.analyze(df_raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    res = out["results"].copy()
    res["timestamp"] = res["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    run_id = uuid.uuid4().hex[:12]
    blocked = store.blocked_set()
    alerts = _records(res)
    for a in alerts:
        a.update(run_id=run_id, status="blocked" if a["tag_id"] in blocked else "open",
                 note="", updated_at=None)

    summary = out["summary"]
    summary["by_day"] = {k: int(v) for k, v in res["timestamp"].str[:10].value_counts().sort_index().items()}
    run = {"run_id": run_id, "created_at": now_iso(), "filename": filename,
           "summary": summary, "notes": out["notes"], "evaluation": out.get("evaluation")}
    store.save_run(run, alerts)
    run["status_counts"] = store.status_counts(run_id)
    return run


@app.get("/api/health")
def health():
    return {"status": "ok", "storage": state["store"].name,
            "models_loaded": sorted(state["detector"].models), "runs": len(state["store"].list_runs(1000))}


@app.post("/api/analyze")
def analyze(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    content = file.file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB.")
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(status_code=400, detail="The CSV has no rows.")
    return _run_analysis(df, file.filename)


@app.post("/api/analyze-demo")
def analyze_demo():
    return _run_analysis(pd.read_csv(cfg.DEMO_CSV), cfg.DEMO_CSV.name)


@app.get("/api/runs")
def list_runs(limit: int = Query(20, ge=1, le=100)):
    return state["store"].list_runs(limit)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    store = state["store"]
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return {**run, "status_counts": store.status_counts(run_id)}


@app.get("/api/runs/{run_id}/alerts")
def get_alerts(run_id: str,
               severity: Optional[Literal["Critical", "Warning", "Info"]] = None,
               detector: Optional[str] = None,
               plaza: Optional[str] = None,
               status: Optional[Status] = None,
               q: Optional[str] = Query(None, max_length=60),
               min_score: Optional[float] = Query(None, ge=0, le=100),
               page: int = Query(1, ge=1),
               page_size: int = Query(50, ge=1, le=200)):
    store = state["store"]
    if not store.get_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    flt = {"severity": severity, "detector": detector, "plaza": plaza, "status": status,
           "q": q, "min_score": min_score}
    total, items = store.query_alerts(run_id, flt, page, page_size)
    return {"total": total, "page": page, "page_size": page_size, "items": items}


class AlertUpdate(BaseModel):
    status: Status
    note: str = Field("", max_length=500)


@app.patch("/api/runs/{run_id}/alerts/{txn_id}")
def update_alert(run_id: str, txn_id: str, body: AlertUpdate):
    store = state["store"]
    alert = store.update_alert(run_id, txn_id, body.status, body.note)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")
    cascaded = 0
    if body.status == "blocked":
        store.block_tag(alert["tag_id"], run_id, txn_id)
        cascaded = store.cascade_block(run_id, alert["tag_id"])
    return {"alert": alert, "other_alerts_blocked": cascaded}


@app.get("/api/blocked-tags")
def blocked_tags():
    return state["store"].blocked_tags()


@app.delete("/api/blocked-tags/{tag_id}")
def unblock(tag_id: str):
    if not state["store"].unblock_tag(tag_id):
        raise HTTPException(status_code=404, detail="Tag is not blocked.")
    return {"unblocked": tag_id}


@app.get("/api/metrics")
def metrics():
    if not cfg.METRICS_JSON.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found. Run python ml/train.py")
    return json.loads(cfg.METRICS_JSON.read_text())


@app.get("/api/diagnostics")
def diagnostics():
    """Model-selection leaderboard and over/underfitting evidence (written by the ML pipeline)."""
    if not (cfg.SELECTION_JSON.exists() and cfg.DIAGNOSTICS_JSON.exists()):
        raise HTTPException(status_code=404, detail="Diagnostics not found. Run python ml/run_pipeline.py")
    return {"selection": json.loads(cfg.SELECTION_JSON.read_text()),
            "diagnostics": json.loads(cfg.DIAGNOSTICS_JSON.read_text())}


@app.get("/api/plazas")
def plazas():
    return _records(state["detector"].plazas)


if cfg.FRONTEND_DIR.exists():                       # mounted last so /api/* wins
    app.mount("/", StaticFiles(directory=str(cfg.FRONTEND_DIR), html=True), name="frontend")