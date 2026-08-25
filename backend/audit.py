"""
Append-only audit trail. Every decision and action gets a timestamped entry, so
the entire recovery workflow is explainable and verifiable end-to-end - the
property Track 03 prizes ("verification, not generation, is the bottleneck").
"""

from datetime import datetime


class AuditTrail:
    def __init__(self):
        self.entries = []

    def log(self, payment_id: str, step: str, detail: dict):
        self.entries.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "payment_id": payment_id,
            "step": step,
            "detail": detail,
        })

    def for_payment(self, payment_id: str):
        return [e for e in self.entries if e["payment_id"] == payment_id]
