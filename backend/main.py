"""
RetryBrain API - the full detect -> diagnose -> score -> decide -> bounded-recovery
pipeline, plus the measured-results endpoints and the dashboard.

Run:  uvicorn backend.main:app --reload
Docs: http://127.0.0.1:8000/docs      Dashboard: http://127.0.0.1:8000/
"""

import os
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.audit import AuditTrail
from backend.diagnosis import diagnose
from backend import store
from backend.runner import recover_one, run_batch

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
DASHBOARD = os.path.join(REPO_ROOT, "frontend", "index.html")

app = FastAPI(title="RetryBrain", version="1.0.0",
              description="Detect revenue at risk, decide the intervention, "
                          "execute a bounded recovery workflow, and measure the money recovered.")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PaymentFailedEvent(BaseModel):
    payment_id: str
    customer_id: str
    amount: float
    currency: str = "INR"
    method: str
    issuing_bank: str
    failure_code: str
    timestamp: str
    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None
    attempt_number: int = 1
    in_downtime_window: bool = False
    past_success_rate: Optional[float] = None
    dnd: bool = False
    consent_whatsapp: bool = True


@app.on_event("startup")
def _warm_start():
    """Load the last snapshot so the dashboard has data instantly; if there's no
    snapshot but a batch file exists, run the batch once."""
    if not store.load():
        try:
            run_batch()
        except FileNotFoundError:
            pass  # no batch.json yet - dashboard will show an empty state


def _fill_time_fields(record: dict) -> dict:
    """Derive hour_of_day / day_of_week from the timestamp when the caller omits them."""
    if record.get("hour_of_day") is None or record.get("day_of_week") is None:
        try:
            dt = datetime.fromisoformat(record["timestamp"])
        except (ValueError, KeyError):
            dt = datetime.now()
        record["hour_of_day"] = dt.hour if record.get("hour_of_day") is None else record["hour_of_day"]
        record["day_of_week"] = dt.weekday() if record.get("day_of_week") is None else record["day_of_week"]
    return record


@app.get("/health")
def health():
    return {"status": "ok", "has_results": bool(store.results), "payments": len(store.results)}


@app.post("/events/payment-failed")
def ingest_payment_failed(event: PaymentFailedEvent):
    """Run the FULL bounded recovery workflow for a single failed payment and
    return the diagnosis, the outcome, and this payment's audit trail."""
    record = _fill_time_fields(event.model_dump())
    record["root_cause"] = diagnose(record["failure_code"])
    trail = AuditTrail()
    outcome = recover_one(record, trail)
    return {
        "diagnosis": record["root_cause"],
        "outcome": outcome,
        "audit": trail.for_payment(record["payment_id"]),
    }


@app.post("/run-batch")
def run_batch_endpoint():
    """Recover the whole batch and return the measured scoreboard."""
    try:
        return run_batch()
    except FileNotFoundError:
        raise HTTPException(404, "data/batch.json not found - run `python data/generate.py` first")


@app.get("/metrics")
def metrics():
    if not store.metrics:
        try:
            return run_batch()
        except FileNotFoundError:
            raise HTTPException(404, "No metrics yet - generate data and run the batch")
    return store.metrics


@app.get("/results")
def results():
    return store.results


@app.get("/audit/{payment_id}")
def audit(payment_id: str):
    entries = store.audit_for(payment_id)
    if not entries:
        raise HTTPException(404, f"no audit trail for {payment_id}")
    return entries


@app.get("/")
def dashboard():
    if os.path.exists(DASHBOARD):
        return FileResponse(DASHBOARD)
    raise HTTPException(404, "dashboard not built yet (frontend/index.html missing)")
