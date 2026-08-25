"""
Batch runner - the orchestration that actually recovers money across the batch
and measures it. This is where Track 03's "bounded recovery workflow" lives.

For each failed payment we loop: diagnose -> score the retry we WOULD perform ->
decide -> execute -> (retry again | escalate along the compliance ladder | stop).
The loop is BOUNDED three ways: the retry cap (MAX_ATTEMPTS), the escalation
ladder ending in human_handoff, and a hard workflow guard. Every step is logged
to the shared audit trail.

We also run a NAIVE baseline (one blind immediate retry per payment) so the
report can prove RetryBrain recovers more than a dumb retry would. Both use the
same seeded oracle, so runs reproduce exactly.

Run:  python -m backend.runner
"""

import os
import json

from backend.diagnosis import diagnose
from backend.decision_engine import decide, optimal_retry, MAX_ATTEMPTS
from backend.agent.agent import run_recovery
from backend.agent.compliance import next_escalation
from backend.model import infer
from backend import simulator, store
from backend.metrics import build_metrics

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
BATCH_PATH = os.path.join(REPO_ROOT, "data", "batch.json")

GUARD = 12   # hard cap on workflow steps per payment (belt-and-suspenders)


def _retry_conditioned_score(ev: dict) -> float:
    """Score the retry we are ACTUALLY going to perform (at its optimal time),
    not the past failed attempt. The model is trained on failure-time features,
    but the decision should be gated on the outcome of the *planned* action:
      * insufficient_funds -> scored at the morning retry hour (earns the top-up lift)
      * bank_downtime      -> scored AFTER the outage window (in_downtime = 0)
    This is the correct way to use a predictive model inside a policy, and it keeps
    the ML score - not a hard-coded rule - in charge of retry-vs-nudge."""
    _, hour, _dt_over = optimal_retry(ev)
    shadow = dict(ev)
    shadow["hour_of_day"] = int(hour)
    if ev["failure_code"] == "bank_downtime":
        shadow["in_downtime_window"] = 0
    return infer.score(shadow)


def _outcome(ev, recovered, amount_recovered, final_action, p, retries, contacts, note):
    return {
        "payment_id": ev["payment_id"],
        "customer_id": ev.get("customer_id"),
        "failure_code": ev["failure_code"],
        "root_cause": (ev.get("root_cause") or {}).get("cause"),
        "amount": round(float(ev["amount"]), 2),
        "method": ev.get("method"),
        "recovered": bool(recovered),
        "amount_recovered": round(float(amount_recovered), 2),
        "final_action": final_action,
        "p_success": round(float(p or 0.0), 3),
        "retries_used": retries,
        "contacts_used": contacts,
        "dnd": bool(ev.get("dnd", False)),
        "note": note,
    }


def recover_one(event: dict, audit) -> dict:
    """Run one bounded recovery workflow for a single failed payment."""
    ev = dict(event)
    ev["attempt_number"] = int(ev.get("attempt_number", 1))
    pid = ev["payment_id"]
    amount = float(ev["amount"])
    escalation = None
    retries_used = 0
    contacts_used = 0
    first_p = None
    last_action = None

    for _ in range(GUARD):
        ev["root_cause"] = diagnose(ev["failure_code"])
        p = _retry_conditioned_score(ev)
        first_p = p if first_p is None else first_p
        decision = decide(ev, p)
        last_action = decision.action

        # --- stopping rule fired in the policy (e.g. attempt cap) ---
        if decision.action == "stop":
            run_recovery(ev, decision, audit)          # logs the stop + reason
            return _outcome(ev, False, 0.0, "stop", first_p, retries_used,
                            contacts_used, "stopped: " + decision.reason)

        # --- retry track: bounded by MAX_ATTEMPTS via attempt_number ---
        if decision.action == "retry":
            res = run_recovery(ev, decision, audit)
            retries_used += 1
            if res["recovered"]:
                return _outcome(ev, True, amount, "retry", first_p, retries_used,
                                contacts_used, res["note"])
            ev["attempt_number"] += 1                  # consume an attempt; cap -> stop next loop
            continue

        # --- contact track: bounded by the escalation ladder -> human_handoff ---
        if decision.action in ("dun", "switch_method"):
            escalation = next_escalation(escalation)
            if escalation is None or escalation == "human_handoff":
                audit.log(pid, "human_handoff", {"reason": "escalation ladder exhausted"})
                return _outcome(ev, False, 0.0, decision.action, first_p, retries_used,
                                contacts_used, "unrecovered -> escalated to human handoff")
            audit.log(pid, "escalate", {"level": escalation})
            res = run_recovery(ev, decision, audit)
            contacts_used += 1
            if res["recovered"]:
                return _outcome(ev, True, amount, decision.action, first_p, retries_used,
                                contacts_used, res["note"])
            if "could not contact" in res["note"]:     # compliance block (DND) -> hand off
                audit.log(pid, "human_handoff", {"reason": res["note"]})
                return _outcome(ev, False, 0.0, decision.action, first_p, retries_used,
                                contacts_used, "human handoff (" + res["note"] + ")")
            continue

        # --- any other terminal action ---
        run_recovery(ev, decision, audit)
        return _outcome(ev, False, 0.0, decision.action, first_p, retries_used,
                        contacts_used, "escalated / unresolved")

    return _outcome(ev, False, 0.0, last_action, first_p, retries_used,
                    contacts_used, "workflow guard reached")


def _baseline_recovered(batch: list) -> list:
    """Naive baseline: one blind immediate retry per payment, at the failure hour
    and the failure's own downtime state - no timing, no method switch, no nudge."""
    simulator.reseed(7)
    return [bool(simulator.simulate_retry(ev, int(ev["hour_of_day"]),
                                          in_downtime_window=bool(ev["in_downtime_window"])))
            for ev in batch]


def run_batch(path: str = None) -> dict:
    """Recover the whole batch, measure it vs the baseline, store + snapshot."""
    path = path or BATCH_PATH
    with open(path) as f:
        batch = json.load(f)

    store.reset()
    simulator.reseed(7)                                # reproducible RetryBrain phase
    rb_results = [recover_one(ev, store.audit) for ev in batch]
    baseline = _baseline_recovered(batch)              # reproducible baseline phase

    metrics = build_metrics(rb_results, baseline, batch)
    metrics["model"] = "trained" if os.path.exists(infer.MODEL_PATH) else "heuristic"
    store.save(rb_results, metrics)
    return metrics


def _print_summary(m: dict):
    print("=" * 64)
    print(f"RetryBrain batch run   (scoring: {m['model']})")
    print("=" * 64)
    print(f"Payments in batch      : {m['total_payments']}")
    print(f"Rupees at risk         : Rs.{m['rupees_at_risk']:,.2f}")
    print("-" * 64)
    print(f"RetryBrain recovered   : {m['recovered_count']}/{m['total_payments']} "
          f"({m['recovery_rate']*100:.1f}%)  Rs.{m['rupees_recovered']:,.2f}")
    print(f"Naive baseline         : {m['baseline']['recovered_count']}/{m['total_payments']} "
          f"({m['baseline']['recovery_rate']*100:.1f}%)  Rs.{m['baseline']['rupees_recovered']:,.2f}")
    print(f"UPLIFT                 : +{m['uplift']['extra_payments_recovered']} payments, "
          f"+Rs.{m['uplift']['rupees']:,.2f}, +{m['uplift']['recovery_rate_points']} pts")
    print("-" * 64)
    print("Per failure code (recovered / attempted, baseline):")
    for code, c in m["by_failure_code"].items():
        print(f"  {code:20s} {c['recovered']:>2}/{c['attempted']:<2}  "
              f"base {c['baseline_recovered']:>2}   Rs.{c['rupees_recovered']:,.2f}")
    print("-" * 64)
    print(f"Exceptions (not recovered): {m['exception_count']}")
    for e in m["exceptions"][:8]:
        print(f"  {e['payment_id']}  {e['failure_code']:18s} Rs.{e['amount']:<9.2f} {e['reason']}")
    if m["exception_count"] > 8:
        print(f"  ... and {m['exception_count'] - 8} more")
    print("=" * 64)
    ok = m["recovered_count"] >= m["baseline"]["recovered_count"]
    print("RESULT: RetryBrain " + ("BEATS" if ok else "DID NOT BEAT") + " the naive baseline.")


if __name__ == "__main__":
    _print_summary(run_batch())
