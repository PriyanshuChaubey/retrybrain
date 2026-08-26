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
from backend.decision_engine import decide, optimal_retry, MAX_ATTEMPTS, RETRY_THRESHOLD
from backend.agent.agent import run_recovery
from backend.agent.compliance import next_escalation
from backend.audit import AuditTrail
from backend.model import infer
from backend import simulator, store, economics
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


def _outcome(ev, recovered, amount_recovered, final_action, p, retries, contacts, channel, note):
    cost = economics.workflow_cost(retries, contacts, channel)
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
        "channel": cost["channel"],
        "retry_spent": cost["retry_spent"],
        "outreach_spent": cost["outreach_spent"],
        "cost_spent": cost["cost_spent"],
        "dnd": bool(ev.get("dnd", False)),
        "note": note,
    }


def recover_one(event: dict, audit, budget=None) -> dict:
    """Run one bounded recovery workflow for a single failed payment.

    `budget` is an economics.Budget governing PAID outreach only (retries are free
    and never gated). The default — an unlimited budget — funds every contact, so
    the headline demo shows RetryBrain's full capability, unchanged. A finite budget
    turns this into an allocation: only failures the allocator funded (highest
    expected ROI first) get contacted; the rest fall through to the free retry and
    are listed honestly as exceptions."""
    budget = budget or economics.Budget.unlimited()
    ev = dict(event)
    ev["attempt_number"] = int(ev.get("attempt_number", 1))
    pid = ev["payment_id"]
    amount = float(ev["amount"])
    escalation = None
    retries_used = 0
    contacts_used = 0
    channel = None
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
                            contacts_used, channel, "stopped: " + decision.reason)

        # --- retry track: bounded by MAX_ATTEMPTS via attempt_number (free) ---
        if decision.action == "retry":
            res = run_recovery(ev, decision, audit)
            retries_used += 1
            if res["recovered"]:
                return _outcome(ev, True, amount, "retry", first_p, retries_used,
                                contacts_used, channel, res["note"])
            ev["attempt_number"] += 1                  # consume an attempt; cap -> stop next loop
            continue

        # --- contact track: bounded by the escalation ladder -> human_handoff ---
        if decision.action in ("dun", "switch_method"):
            escalation = next_escalation(escalation)
            if escalation is None or escalation == "human_handoff":
                audit.log(pid, "human_handoff", {"reason": "escalation ladder exhausted"})
                return _outcome(ev, False, 0.0, decision.action, first_p, retries_used,
                                contacts_used, channel, "unrecovered -> escalated to human handoff")

            # A DND / no-consent block is a COMPLIANCE stop, decided before any
            # economics: we never spend budget on a customer we're not allowed to reach.
            if ev.get("dnd"):
                res = run_recovery(ev, decision, audit)   # logs compliance_block; sends nothing
                audit.log(pid, "human_handoff", {"reason": res["note"]})
                return _outcome(ev, False, 0.0, decision.action, first_p, retries_used,
                                contacts_used, channel, "human handoff (" + res["note"] + ")")

            # --- ECONOMIC GATE: is paid outreach for this payment worth funding? ---
            ch = economics.contact_channel(ev)
            cost = economics.ACTION_COST[ch]
            if not budget.is_funded(pid):
                audit.log(pid, "budget_skip",
                          {"reason": "outreach not funded: lower expected ROI than funded failures",
                           "expected_roi": round(economics.expected_roi(ev), 2)})
                return _outcome(ev, False, 0.0, "budget_skip", first_p, retries_used,
                                contacts_used, channel,
                                "outreach skipped: below the ROI budget line (higher-value failures funded first)")
            if not budget.try_spend(cost):
                audit.log(pid, "budget_exhausted",
                          {"reason": "outreach budget already spent on higher-ROI failures"})
                return _outcome(ev, False, 0.0, "budget_exhausted", first_p, retries_used,
                                contacts_used, channel,
                                "outreach budget exhausted before reaching this failure")

            channel = ch
            audit.log(pid, "escalate", {"level": escalation})
            res = run_recovery(ev, decision, audit)
            contacts_used += 1
            if res["recovered"]:
                return _outcome(ev, True, amount, decision.action, first_p, retries_used,
                                contacts_used, channel, res["note"])
            continue

        # --- any other terminal action ---
        run_recovery(ev, decision, audit)
        return _outcome(ev, False, 0.0, decision.action, first_p, retries_used,
                        contacts_used, channel, "escalated / unresolved")

    return _outcome(ev, False, 0.0, last_action, first_p, retries_used,
                    contacts_used, channel, "workflow guard reached")


def _baseline_recovered(batch: list) -> list:
    """Naive baseline: one blind immediate retry per payment, at the failure hour
    and the failure's own downtime state - no timing, no method switch, no nudge."""
    simulator.reseed(7)
    return [bool(simulator.simulate_retry(ev, int(ev["hour_of_day"]),
                                          in_downtime_window=bool(ev["in_downtime_window"])))
            for ev in batch]


def _run_pass(batch: list, budget, audit) -> list:
    """One reproducible RetryBrain sweep over a batch under a given budget.
    The caller reseeds the simulator so passes are comparable."""
    return [recover_one(ev, audit, budget=budget) for ev in batch]


def run_batch(path: str = None, batch: list = None) -> dict:
    """Recover the whole batch, measure it vs the baseline, store + snapshot.

    The headline run uses an UNLIMITED budget (full capability, numbers unchanged);
    the economics block then adds the cost/ROI view and the budget-constrained
    efficient frontier on top, without touching the headline."""
    if batch is None:
        path = path or BATCH_PATH
        with open(path) as f:
            batch = json.load(f)

    store.reset()
    simulator.reseed(7)                                # reproducible RetryBrain phase
    rb_results = _run_pass(batch, economics.Budget.unlimited(), store.audit)
    baseline = _baseline_recovered(batch)              # reproducible baseline phase

    metrics = build_metrics(rb_results, baseline, batch)
    metrics["model"] = "trained" if os.path.exists(infer.MODEL_PATH) else "heuristic"
    metrics["economics"] = build_economics(batch, rb_results)
    store.save(rb_results, metrics)
    return metrics


# ---------------------------------------------------------------------------
# Economics: cost/ROI of the headline run + the budget-constrained frontier.
# ---------------------------------------------------------------------------

def _primary_path(ev: dict, score: float) -> str:
    """Mirror decision_engine.decide() for a fresh (attempt 1) failure: does the
    policy RETRY this payment (free) or CONTACT the customer (paid outreach)?"""
    hint = (ev.get("root_cause") or {}).get("suggested_action")
    if ev["failure_code"] == "expired_card" or hint == "request_new_method":
        return "contact"
    return "retry" if score >= RETRY_THRESHOLD else "contact"


def _expected_frontier(batch: list, steps: int = 12):
    """Analytic 'efficient frontier': EXPECTED rupees recovered as the outreach
    budget grows from 0 to fully funded. Smooth and monotonic by construction
    because outreach is funded in the allocator's own descending-ROI order.

    Expected recovery = a fixed retry floor (money the free retries recover, using
    the model's own P(retry succeeds)) + the funded contacts' expected value (the
    documented per-cause conversion prior x amount). It uses only priors known at
    decision time, never the simulator's hidden ground truth."""
    retry_floor = 0.0
    candidates = []   # (roi, cost, expected_contact_value_gross) for contact-path payments
    for ev in batch:
        e = dict(ev)
        e["attempt_number"] = 1
        e["root_cause"] = diagnose(e["failure_code"])
        score = _retry_conditioned_score(e)
        amount = float(e["amount"])
        if _primary_path(e, score) == "retry":
            retry_floor += score * amount                      # bounded free retries
        elif not e.get("dnd") and economics.expected_contact_value(e) > 0:
            candidates.append((economics.expected_roi(e),
                               economics.contact_cost(e),
                               economics.conversion_prior(e) * amount))
        # DND / negative-EV contact payments contribute ~0 automated recovery

    candidates.sort(key=lambda t: t[0], reverse=True)           # highest ROI first
    full_outreach = sum(c for _, c, _ in candidates)

    grid = [full_outreach * i / steps for i in range(steps + 1)] if full_outreach > 0 else [0.0]
    frontier = []
    for b in grid:
        running, recovered, funded = 0.0, retry_floor, 0
        for _roi, cost, gross in candidates:
            if running + cost <= b + 1e-9:
                recovered += gross
                running += cost
                funded += 1
        frontier.append({"budget": round(b, 2),
                         "expected_recovered": round(recovered, 2),
                         "funded_contacts": funded,
                         "outreach_cost": round(running, 2)})
    return frontier, full_outreach


def _measure_budgeted(batch: list, budget_limit: float) -> dict:
    """Actually RUN the workflow under a finite budget (throwaway audit, no store
    writes) so we can report a MEASURED point, not just an expected one."""
    simulator.reseed(7)
    budget = economics.Budget(budget_limit).plan(batch)
    results = _run_pass(batch, budget, AuditTrail())
    rec = [r for r in results if r["recovered"]]
    return {
        "budget": round(float(budget_limit), 2),
        "recovered_count": len(rec),
        "rupees_recovered": round(sum(r["amount_recovered"] for r in rec), 2),
        "outreach_cost": round(sum(r["outreach_spent"] for r in results), 2),
        "contacts": sum(r["contacts_used"] for r in results),
    }


def build_economics(batch: list, rb_results: list) -> dict:
    """Cost/ROI of the (unlimited) headline run + the budget-constrained frontier."""
    retry_spent = sum(r["retry_spent"] for r in rb_results)
    outreach_spent = sum(r["outreach_spent"] for r in rb_results)
    total_cost = retry_spent + outreach_spent
    rupees_recovered = sum(r["amount_recovered"] for r in rb_results)
    rb_recovered_count = sum(1 for r in rb_results if r["recovered"])
    roi = (rupees_recovered / total_cost) if total_cost > 0 else 0.0

    frontier, full_outreach = _expected_frontier(batch)
    max_expected = frontier[-1]["expected_recovered"] if frontier else 0.0
    # "efficient budget" = smallest sweep point capturing >=95% of the max expected recovery
    eff = next((pt for pt in frontier if pt["expected_recovered"] >= 0.95 * max_expected),
               frontier[-1] if frontier else {"budget": 0.0, "expected_recovered": 0.0})
    measured = _measure_budgeted(batch, eff["budget"]) if full_outreach > 0 else None

    # MEASURED free-retry floor: run the identical pipeline with the outreach budget
    # set to Rs.0 (retries only, no paid contact). This is the honest, consistent-
    # denominator way to isolate what outreach actually buys on top of free retries.
    floor = _measure_budgeted(batch, 0.0) if full_outreach > 0 else None
    lift_rupees = round(rupees_recovered - floor["rupees_recovered"], 2) if floor else 0.0
    lift_payments = rb_recovered_count - floor["recovered_count"] if floor else 0
    outreach_roi = round(lift_rupees / outreach_spent, 0) if outreach_spent > 0 else 0.0

    return {
        "action_costs": economics.ACTION_COST,
        "conversion_prior": economics.CONTACT_CONVERSION_PRIOR,
        "measured_run": {
            "retry_spent": round(retry_spent, 2),
            "outreach_spent": round(outreach_spent, 2),
            "total_cost": round(total_cost, 2),
            "contacts": sum(r["contacts_used"] for r in rb_results),
            "recovered_count": rb_recovered_count,
            "rupees_recovered": round(rupees_recovered, 2),
            "roi": round(roi, 1),                       # Rs recovered per Re spent (total)
        },
        # Free retries alone (Rs.0 outreach) vs the incremental lift outreach buys.
        "measured_floor": floor,
        "outreach_lift": {
            "rupees": lift_rupees,
            "payments": lift_payments,
            "outreach_cost": round(outreach_spent, 2),
            "roi_per_rupee": outreach_roi,              # Rs recovered per Re of OUTREACH
        },
        "full_outreach_cost": round(full_outreach, 2),
        "frontier": frontier,
        "efficient_budget": {
            "budget": round(eff["budget"], 2),
            "pct_of_full_outreach": round(100 * eff["budget"] / full_outreach, 1) if full_outreach > 0 else 0.0,
            "pct_recovery_captured": round(100 * eff["expected_recovered"] / max_expected, 1) if max_expected > 0 else 0.0,
            # Measured against the SAME denominators the reader sees in measured_run:
            "pct_of_measured_outreach": round(100 * measured["outreach_cost"] / outreach_spent, 1)
                                        if (measured and outreach_spent > 0) else None,
            "pct_of_measured_recovery": round(100 * measured["rupees_recovered"] / rupees_recovered, 1)
                                        if (measured and rupees_recovered > 0) else None,
            "measured": measured,
        },
    }


def robustness(seeds=(42, 7, 13, 99, 2024), n: int = 60) -> dict:
    """Anti-cherry-pick: regenerate the batch under several seeds and report the
    uplift's mean and range, so the headline can't be a single lucky draw.

    We mirror data/generate.py's main() order exactly (seed -> customers ->
    history -> batch), so seed 42 reproduces the shipped data/batch.json and the
    other seeds are honest independent draws from the same generator."""
    import random as _random
    from data import generate as gen

    runs = []
    for s in seeds:
        _random.seed(s)
        customers = [gen._make_customer(i) for i in range(120)]
        gen.generate_history(customers, 1500)          # consume RNG exactly like main()
        b = gen.generate_batch(customers, n)
        simulator.reseed(7)
        rb = _run_pass(b, economics.Budget.unlimited(), AuditTrail())
        base = _baseline_recovered(b)
        m = build_metrics(rb, base, b)
        runs.append({
            "seed": s,
            "rb_rate": round(m["recovery_rate"] * 100, 1),
            "baseline_rate": round(m["baseline"]["recovery_rate"] * 100, 1),
            "uplift_pts": m["uplift"]["recovery_rate_points"],
            "uplift_rupees": m["uplift"]["rupees"],
        })
    ups = [r["uplift_pts"] for r in runs]
    rup = [r["uplift_rupees"] for r in runs]
    return {
        "seeds": list(seeds),
        "runs": runs,
        "uplift_pts_mean": round(sum(ups) / len(ups), 1),
        "uplift_pts_min": min(ups),
        "uplift_pts_max": max(ups),
        "uplift_rupees_mean": round(sum(rup) / len(rup), 2),
        "uplift_rupees_min": min(rup),
        "uplift_rupees_max": max(rup),
    }


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


def _print_economics(m: dict):
    e = m.get("economics")
    if not e:
        return
    r = e["measured_run"]
    eb = e["efficient_budget"]
    floor = e.get("measured_floor")
    lift = e.get("outreach_lift", {})
    print("=" * 64)
    print("RECOVERY ECONOMICS  (retries are free; only customer outreach is budgeted)")
    print("=" * 64)
    if floor:
        print(f"Free retries alone     : {floor['recovered_count']} recovered, "
              f"Rs.{floor['rupees_recovered']:,.2f} at Rs.0.00 outreach")
    print(f"+ compliant outreach   : {r['recovered_count']} recovered, "
          f"Rs.{r['rupees_recovered']:,.2f} (outreach Rs.{r['outreach_spent']:,.2f} on {r['contacts']} messages)")
    if lift:
        print(f"Outreach lift          : +{lift['payments']} payments, +Rs.{lift['rupees']:,.2f} "
              f"for Rs.{lift['outreach_cost']:,.2f}  ->  Rs.{lift['roi_per_rupee']:,.0f} recovered per Re.1 of outreach")
    print(f"Total workflow spend   : Rs.{r['total_cost']:,.2f} "
          f"(Rs.{r['retry_spent']:,.2f} retries + Rs.{r['outreach_spent']:,.2f} outreach)  ->  "
          f"ROI Rs.{r['roi']:,.1f} per Re.1")
    print("-" * 64)
    print("Expected efficient frontier (priors, no oracle): funding the highest-ROI")
    print(f"  nudges first, the smallest budget capturing >=95% of the EXPECTED max is Rs.{eb['budget']:,.2f}.")
    mb = eb.get("measured")
    if mb:
        pr = eb.get("pct_of_measured_recovery")
        ps = eb.get("pct_of_measured_outreach")
        tail = ""
        if pr is not None and ps is not None:
            tail = f" ({pr:.0f}% of the fully-funded revenue for {ps:.0f}% of the full outreach spend)"
        print(f"  Measured at that budget: {mb['recovered_count']} recovered, "
              f"Rs.{mb['rupees_recovered']:,.2f}{tail}.")


def _print_robustness(rb: dict):
    print("=" * 64)
    print("ROBUSTNESS  (same policy, batch regenerated across seeds - anti-cherry-pick)")
    print("=" * 64)
    for r in rb["runs"]:
        print(f"  seed {r['seed']:<5} RB {r['rb_rate']:>5.1f}%  base {r['baseline_rate']:>5.1f}%  "
              f"uplift +{r['uplift_pts']:>4.1f} pts  +Rs.{r['uplift_rupees']:,.2f}")
    print("-" * 64)
    print(f"Uplift across seeds    : +{rb['uplift_pts_mean']:.1f} pts mean "
          f"(range +{rb['uplift_pts_min']:.1f} to +{rb['uplift_pts_max']:.1f})")
    print(f"Rupees uplift          : +Rs.{rb['uplift_rupees_mean']:,.2f} mean "
          f"(range +Rs.{rb['uplift_rupees_min']:,.2f} to +Rs.{rb['uplift_rupees_max']:,.2f})")


if __name__ == "__main__":
    import sys
    m = run_batch()
    _print_summary(m)
    _print_economics(m)
    if "--robust" in sys.argv:
        _print_robustness(robustness())
