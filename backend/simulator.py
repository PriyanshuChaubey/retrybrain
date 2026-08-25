"""
Payment simulator - the "real world" ground truth the model is trying to
approximate. Given a recovery ACTION, it returns whether the money came back.

This mirrors the domain logic in data/generate.py (timing matters; expired cards
don't recover on a blind retry; bank_downtime only recovers after the window).
Keeping the outcome HERE - separate from the model - is what lets us honestly
MEASURE recovery: the model never sees these rules directly.
"""

import random

_rng = random.Random(7)  # seeded -> reproducible batch runs and demos


def reseed(seed: int = 7):
    """Reset the oracle RNG so a batch run reproduces exactly (the runner calls this
    before each phase so RetryBrain and the baseline are measured under the same luck)."""
    _rng.seed(seed)

BASE_RECOVERY = {
    "insufficient_funds": 0.45,
    "do_not_honor": 0.25,
    "bank_downtime": 0.10,
    "expired_card": 0.02,
    "network_error": 0.80,
    "3ds_failure": 0.35,
    "other": 0.30,
}

DUNNING_SUCCESS = 0.30   # chance a customer updates method / pays after a nudge


def retry_success_prob(failure_code, retry_hour, attempt_number, in_downtime_window, method):
    p = BASE_RECOVERY[failure_code]
    if failure_code == "insufficient_funds" and 6 <= retry_hour <= 11:
        p += 0.25                                   # morning top-ups
    elif failure_code == "bank_downtime":
        p = 0.85 if not in_downtime_window else 0.10  # only after the window
    elif failure_code == "network_error" and attempt_number == 1:
        p += 0.10
    p -= 0.08 * (attempt_number - 1)                # diminishing returns
    if method == "upi":
        p += 0.03
    return max(0.0, min(1.0, p))


def simulate_retry(event, retry_hour, in_downtime_window=None):
    idw = bool(event.get("in_downtime_window")) if in_downtime_window is None else bool(in_downtime_window)
    p = retry_success_prob(event["failure_code"], int(retry_hour),
                           int(event["attempt_number"]), idw, event["method"])
    return _rng.random() < p


def simulate_dunning(event):
    """Customer responds to a compliant dunning nudge and pays."""
    return _rng.random() < DUNNING_SUCCESS
