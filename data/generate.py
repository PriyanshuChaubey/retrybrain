"""
RetryBrain - synthetic data generator (Track 03: AI Revenue Recovery).

Produces (next to this file, in data/):
  history.csv  - labeled past failed-payment retries for TRAINING the
                 retry-success model (includes the retry_succeeded label).
  batch.json   - a fresh batch of >=50 NEW failed payments (no outcome yet)
                 for the agent to detect -> diagnose -> decide -> recover.

Design note (say this in your interview): this is a *ground-truth* generator.
It encodes the "true" recovery probability for each failure type and timing, so
(a) the ML model has real signal to learn, and (b) the simulator can decide
whether a retry actually succeeds. Generating believable data is itself a signal
of domain understanding - the buildathon brief explicitly rewards it.

Pure standard library - no pip install needed.  Run:  python data/generate.py
"""

import os
import csv
import json
import random
import argparse
from datetime import datetime, timedelta

random.seed(42)  # reproducible batches - good for demos and tests

# Fixed reference "now" so the dataset is FULLY reproducible. With only random.seed()
# set, the timestamps below still read the wall clock, so hour_of_day (and therefore
# some labels) drifted from run to run - the same repo produced a different ROC-AUC on
# a different day. Pinning the clock makes generate.py emit byte-identical history.csv
# and batch.json on any machine, any day, so every number in the README stays true.
REFERENCE_NOW = datetime(2026, 1, 15, 12, 0, 0)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- Domain constants -------------------------------------------------------

METHODS = ["card", "upi", "netbanking", "wallet"]
BANKS = ["HDFC", "ICICI", "SBI", "Axis", "Kotak", "Yes", "PNB", "IDFC"]
CITIES = ["Bengaluru", "Mumbai", "Delhi", "Chennai", "Pune", "Hyderabad"]

# Failure codes and how often each occurs (weights).
FAILURE_WEIGHTS = {
    "insufficient_funds": 30,
    "do_not_honor": 15,
    "bank_downtime": 15,
    "expired_card": 10,
    "network_error": 10,
    "3ds_failure": 10,
    "other": 10,
}

# Base probability that a *well-timed* retry recovers the payment, by failure
# type. These encode the domain truth the model should learn.
BASE_RECOVERY = {
    "insufficient_funds": 0.45,
    "do_not_honor": 0.25,
    "bank_downtime": 0.10,   # low DURING downtime; boosted after the window
    "expired_card": 0.02,    # a blind retry ~never works; needs a new method
    "network_error": 0.80,   # transient; a quick retry usually works
    "3ds_failure": 0.35,
    "other": 0.30,
}


def recovery_probability(failure_code, hour, attempt_number,
                         in_downtime_window, method):
    """Ground-truth probability that retrying THIS payment now succeeds.

    Timing is where most of the recovery uplift comes from - the key interview
    talking point. Keep this the single source of truth for both training-data
    labels and the live simulator outcome.
    """
    p = BASE_RECOVERY[failure_code]

    if failure_code == "insufficient_funds":
        if 6 <= hour <= 11:          # people top up in the morning
            p += 0.25
        if REFERENCE_NOW.day <= 3:   # crude "salary window" proxy (fixed for reproducibility)
            p += 0.10
    elif failure_code == "bank_downtime":
        p = 0.85 if not in_downtime_window else 0.10  # recovers once bank is back
    elif failure_code == "network_error":
        if attempt_number == 1:
            p += 0.10

    p -= 0.08 * (attempt_number - 1)  # diminishing returns per attempt
    if method == "upi":
        p += 0.03                     # UPI retries slightly better in India

    return max(0.0, min(1.0, p))


def _weighted_failure():
    codes = list(FAILURE_WEIGHTS.keys())
    weights = list(FAILURE_WEIGHTS.values())
    return random.choices(codes, weights=weights, k=1)[0]


def _make_customer(i):
    return {
        "customer_id": f"cust_{i:04d}",
        "past_success_rate": round(random.uniform(0.55, 0.98), 3),
        "preferred_method": random.choice(METHODS),
        "city": random.choice(CITIES),
        "dnd": random.random() < 0.10,            # 10% opted out of comms
        "consent_whatsapp": random.random() < 0.70,
    }


def _make_event(customer, when, attempt_number):
    failure_code = _weighted_failure()
    method = customer["preferred_method"] if random.random() < 0.7 else random.choice(METHODS)
    in_downtime = failure_code == "bank_downtime" and random.random() < 0.5
    return {
        "payment_id": f"pay_{random.randint(10**9, 10**10 - 1)}",
        "customer_id": customer["customer_id"],
        "amount": round(random.choice([199, 299, 499, 999, 1499, 2999, 4999]) + random.random(), 2),
        "currency": "INR",
        "method": method,
        "issuing_bank": random.choice(BANKS),
        "failure_code": failure_code,
        "timestamp": when.isoformat(),
        "hour_of_day": when.hour,
        "day_of_week": when.weekday(),
        "attempt_number": attempt_number,
        "in_downtime_window": in_downtime,
        "past_success_rate": customer["past_success_rate"],
        "dnd": customer["dnd"],
        "consent_whatsapp": customer["consent_whatsapp"],
    }


def generate_history(customers, n_rows):
    """Labeled rows for TRAINING (includes retry_succeeded)."""
    rows = []
    start = REFERENCE_NOW - timedelta(days=30)
    for _ in range(n_rows):
        cust = random.choice(customers)
        when = start + timedelta(minutes=random.randint(0, 30 * 24 * 60))
        ev = _make_event(cust, when, attempt_number=random.randint(1, 3))
        p = recovery_probability(ev["failure_code"], ev["hour_of_day"],
                                 ev["attempt_number"], ev["in_downtime_window"],
                                 ev["method"])
        ev["retry_succeeded"] = 1 if random.random() < p else 0
        rows.append(ev)
    return rows


def generate_batch(customers, n):
    """A fresh batch of NEW failed payments for the agent to work (no label).

    Failures are spread across the last 24h (not clustered at one instant), so the
    batch spans every hour of the day - no single failure hour can drive the result -
    and it is reproducible via REFERENCE_NOW + the seed."""
    now = REFERENCE_NOW
    return [_make_event(random.choice(customers),
                        now - timedelta(minutes=random.randint(0, 24 * 60)),
                        attempt_number=1) for _ in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", type=int, default=120)
    ap.add_argument("--history", type=int, default=1500, help="training rows")
    ap.add_argument("--batch", type=int, default=60, help=">=50 records for the eval batch")
    args = ap.parse_args()

    customers = [_make_customer(i) for i in range(args.customers)]

    history = generate_history(customers, args.history)
    with open(os.path.join(HERE, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    batch = generate_batch(customers, args.batch)
    with open(os.path.join(HERE, "batch.json"), "w") as f:
        json.dump(batch, f, indent=2)

    recovered = sum(r["retry_succeeded"] for r in history)
    print(f"Wrote data/history.csv  ({len(history)} rows, "
          f"{recovered / len(history):.1%} historically recovered)")
    print(f"Wrote data/batch.json   ({len(batch)} new failed payments to recover)")


if __name__ == "__main__":
    main()
