"""
Store - the small amount of shared state the API needs between requests: the
latest batch results, the computed metrics, and the single AuditTrail that every
recovery workflow logs into. Also snapshots the last run to data/last_run.json so
the dashboard has data immediately on a cold start (no need to re-run the batch).

In-memory by design for the buildathon (one process, reproducible batch). Swapping
this for SQLite later is a drop-in: keep the same save()/load() surface.
"""

import os
import json

from backend.audit import AuditTrail

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
LAST_RUN_PATH = os.path.join(REPO_ROOT, "data", "last_run.json")

# Shared, process-wide state.
audit = AuditTrail()
results: list = []
metrics: dict = {}


def reset():
    """Clear state before a fresh batch run."""
    audit.entries.clear()
    results.clear()
    metrics.clear()


def save(new_results: list, new_metrics: dict):
    """Replace the in-memory results/metrics and snapshot them to disk."""
    results[:] = new_results
    metrics.clear()
    metrics.update(new_metrics)
    persist()


def persist():
    """Write a JSON snapshot of the last run (results + metrics + audit)."""
    payload = {"results": results, "metrics": metrics, "audit": audit.entries}
    os.makedirs(os.path.dirname(LAST_RUN_PATH), exist_ok=True)
    with open(LAST_RUN_PATH, "w") as f:
        json.dump(payload, f, indent=2)


def load() -> bool:
    """Rehydrate state from the last snapshot if present. Returns True on success."""
    if not os.path.exists(LAST_RUN_PATH):
        return False
    with open(LAST_RUN_PATH) as f:
        payload = json.load(f)
    results[:] = payload.get("results", [])
    metrics.clear()
    metrics.update(payload.get("metrics", {}))
    audit.entries[:] = payload.get("audit", [])
    return True


def audit_for(payment_id: str) -> list:
    return audit.for_payment(payment_id)
