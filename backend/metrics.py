"""
Metrics - turn a batch of recovery outcomes into the numbers Track 03 is judged
on: money recovered, recovery rate, a per-failure-code breakdown, an HONEST
exception list, and a head-to-head vs a naive baseline (RetryBrain must win).

Kept pure (no I/O, no globals) so it is trivially unit-testable and defensible:
give it the results, it gives you the scoreboard.
"""

from collections import defaultdict


def _round2(x: float) -> float:
    return round(float(x), 2)


def build_metrics(rb_results: list, baseline_recovered: list, batch: list) -> dict:
    """
    rb_results        : list of per-payment outcome dicts from the RetryBrain workflow
                        (keys: payment_id, failure_code, amount, recovered, note, dnd, ...)
    baseline_recovered: list[bool] aligned with `batch` - did a single blind retry work?
    batch             : the original list of events (for baseline rupee math)
    """
    total = len(rb_results)
    recovered = [r for r in rb_results if r["recovered"]]

    rupees_at_risk = sum(r["amount"] for r in rb_results)
    rupees_recovered = sum(r["amount"] for r in recovered)
    recovery_rate = (len(recovered) / total) if total else 0.0

    # --- naive baseline: one immediate blind retry per payment, no intelligence ---
    base_count = sum(1 for ok in baseline_recovered if ok)
    base_rupees = sum(batch[i]["amount"] for i, ok in enumerate(baseline_recovered) if ok)
    base_rate = (base_count / total) if total else 0.0

    # --- per-failure-code breakdown (RetryBrain vs baseline) ---
    by_code = defaultdict(lambda: {"attempted": 0, "recovered": 0, "rupees_recovered": 0.0,
                                   "baseline_recovered": 0})
    for r in rb_results:
        c = by_code[r["failure_code"]]
        c["attempted"] += 1
        if r["recovered"]:
            c["recovered"] += 1
            c["rupees_recovered"] += r["amount"]
    for i, ok in enumerate(baseline_recovered):
        if ok:
            by_code[batch[i]["failure_code"]]["baseline_recovered"] += 1
    by_code = {k: {**v, "rupees_recovered": _round2(v["rupees_recovered"])}
               for k, v in sorted(by_code.items())}

    # --- honest exception list: everything RetryBrain did NOT recover, with a reason ---
    exceptions = [
        {"payment_id": r["payment_id"], "failure_code": r["failure_code"],
         "amount": r["amount"], "final_action": r.get("final_action"),
         "dnd": r.get("dnd", False), "reason": r["note"]}
        for r in rb_results if not r["recovered"]
    ]

    return {
        "total_payments": total,
        "recovered_count": len(recovered),
        "recovery_rate": round(recovery_rate, 4),
        "rupees_at_risk": _round2(rupees_at_risk),
        "rupees_recovered": _round2(rupees_recovered),
        "baseline": {
            "recovered_count": base_count,
            "recovery_rate": round(base_rate, 4),
            "rupees_recovered": _round2(base_rupees),
        },
        "uplift": {
            "rupees": _round2(rupees_recovered - base_rupees),
            "recovery_rate_points": round((recovery_rate - base_rate) * 100, 2),
            "extra_payments_recovered": len(recovered) - base_count,
        },
        "by_failure_code": by_code,
        "exceptions": exceptions,
        "exception_count": len(exceptions),
    }
