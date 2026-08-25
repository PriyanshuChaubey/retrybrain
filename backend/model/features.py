"""
Feature definitions for the retry-success model. Keep this the SINGLE source of
truth so training (train.py) and live scoring (infer.py) always agree on columns
and types.
"""

CATEGORICAL = ["failure_code", "method", "issuing_bank"]
NUMERIC = ["amount", "hour_of_day", "day_of_week",
           "attempt_number", "in_downtime_window", "past_success_rate"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "retry_succeeded"


def normalize(event: dict) -> dict:
    """Coerce one event dict into the exact feature types the model expects."""
    return {
        "failure_code": str(event["failure_code"]),
        "method": str(event["method"]),
        "issuing_bank": str(event["issuing_bank"]),
        "amount": float(event["amount"]),
        "hour_of_day": int(event["hour_of_day"]),
        "day_of_week": int(event["day_of_week"]),
        "attempt_number": int(event["attempt_number"]),
        "in_downtime_window": int(bool(event["in_downtime_window"])),
        "past_success_rate": float(event.get("past_success_rate") or 0.75),
    }
