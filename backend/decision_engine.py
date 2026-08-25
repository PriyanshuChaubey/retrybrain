"""
Decision engine - the policy that turns a diagnosis + a retry-success score into
an intervention:
    retry (at an optimal time) | dun | switch_method | escalate | stop

Design choices you should OWN (be ready to defend them in the interview):
  * RETRY_THRESHOLD - the recovery-vs-cost lever. Higher = fewer, higher-quality
    retries. Tune it against the batch results.
  * MAX_ATTEMPTS - the hard bound that keeps the workflow from spamming.
  * optimal_retry() - timing is where most recovery uplift comes from.
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

RETRY_THRESHOLD = 0.35   # TODO(you): tune against the batch; justify the value
MAX_ATTEMPTS = 3


@dataclass
class Decision:
    action: str                       # retry | dun | switch_method | escalate | stop
    at_time: Optional[str] = None     # ISO timestamp (when action == "retry")
    retry_hour: Optional[int] = None  # hour-of-day the retry is scheduled for
    downtime_over: Optional[bool] = None
    reason: str = ""


def optimal_retry(event: dict):
    """Return (at_time_iso, retry_hour, downtime_over) for the best retry moment.
    Timing rules are per root cause - this is the core of the uplift."""
    fc = event["failure_code"]
    now = datetime.fromisoformat(event["timestamp"]) if event.get("timestamp") else datetime.now()

    if fc == "insufficient_funds":                 # wait for the morning top-up
        t = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return t.isoformat(), 9, False
    if fc == "bank_downtime":                      # wait for the bank to come back
        t = now + timedelta(hours=2)
        return t.isoformat(), t.hour, True
    if fc == "network_error":                      # transient - retry soon
        t = now + timedelta(minutes=10)
        return t.isoformat(), t.hour, bool(event.get("in_downtime_window"))
    t = now + timedelta(hours=3)                   # conservative default
    return t.isoformat(), t.hour, bool(event.get("in_downtime_window"))


def decide(event: dict, p_success: float) -> Decision:
    # Bounded workflow: never exceed the attempt cap.
    if int(event["attempt_number"]) >= MAX_ATTEMPTS:
        return Decision("stop", reason=f"max attempts ({MAX_ATTEMPTS}) reached")

    # A blind retry cannot fix an expired card - go straight to a new method.
    hint = (event.get("root_cause") or {}).get("suggested_action")
    if event["failure_code"] == "expired_card" or hint == "request_new_method":
        return Decision("switch_method",
                        reason="card expired - retry cannot recover; request a new method")

    # High enough recovery odds -> retry at the optimal moment.
    if p_success >= RETRY_THRESHOLD:
        at_time, hour, dt_over = optimal_retry(event)
        return Decision("retry", at_time=at_time, retry_hour=hour, downtime_over=dt_over,
                        reason=f"p_success={p_success:.2f} >= {RETRY_THRESHOLD}; retry at optimal time")

    # Otherwise don't burn a retry - nudge the customer instead.
    return Decision("dun",
                    reason=f"p_success={p_success:.2f} < {RETRY_THRESHOLD}; nudge instead of retrying")
