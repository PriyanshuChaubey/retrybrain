"""
Bounded LLM recovery agent.

Given a diagnosed + scored event and a Decision, it executes the recovery
workflow within hard bounds and logs every step to the audit trail. "Bounded"
means it respects stopping rules and compliance - it never runs away.

Messaging is TEMPLATE-based today, so the whole thing runs offline. To upgrade
to LLM-written dunning, implement `generate_message` with your provider (read the
key/model from env - see .env.example). The rest of the workflow is unchanged.
"""

from backend.agent import compliance
from backend.simulator import simulate_retry, simulate_dunning

TEMPLATES = {
    "insufficient_funds": ("Hi {name}, your payment of Rs.{amount} didn't go through (low balance). "
                           "We'll retry tomorrow morning - please keep funds ready."),
    "expired_card":       ("Hi {name}, the card on file has expired. Please update your payment method "
                           "to complete your Rs.{amount} payment."),
    "do_not_honor":       ("Hi {name}, your bank declined the Rs.{amount} payment. Try another method "
                           "and we'll help you complete it."),
    "default":            ("Hi {name}, your payment of Rs.{amount} failed. We're here to help you "
                           "complete it."),
}


def generate_message(event: dict) -> str:
    """Template dunning. TODO(you): swap for LLM slot-filling (provider-agnostic)."""
    tmpl = TEMPLATES.get(event["failure_code"], TEMPLATES["default"])
    return tmpl.format(name=event.get("customer_id", "there"), amount=round(float(event["amount"])))


def _result(recovered: bool, amount_recovered: float, note: str) -> dict:
    return {"recovered": recovered, "amount_recovered": round(amount_recovered, 2), "note": note}


def run_recovery(event: dict, decision, audit) -> dict:
    """Execute one bounded recovery workflow and return its outcome."""
    pid = event["payment_id"]
    amount = float(event["amount"])
    audit.log(pid, "decision", {"action": decision.action, "reason": decision.reason})

    # Stopping rule already fired in the decision engine.
    if decision.action == "stop":
        audit.log(pid, "stop", {"reason": decision.reason})
        return _result(False, 0.0, "stopped: " + decision.reason)

    # Retry at the optimal moment.
    if decision.action == "retry":
        audit.log(pid, "schedule_retry", {"at": decision.at_time, "reason": decision.reason})
        # downtime_over=True means the outage window has PASSED, so the retry lands
        # OUTSIDE downtime. The simulator's in_downtime_window flag must therefore be
        # the inverse (only bank_downtime consumes it; other codes ignore it).
        ok = simulate_retry(event, decision.retry_hour,
                            in_downtime_window=not decision.downtime_over)
        audit.log(pid, "retry_result", {"recovered": ok})
        return _result(ok, amount if ok else 0.0, "retry succeeded" if ok else "retry failed")

    # Contact the customer (dun / ask for a new method) - only if compliant.
    if decision.action in ("dun", "switch_method"):
        allowed, why = compliance.can_message(event)
        if not allowed:
            audit.log(pid, "compliance_block", {"reason": why})
            return _result(False, 0.0, f"could not contact ({why})")
        channel = "whatsapp" if event.get("consent_whatsapp") else "email"
        msg = generate_message(event)
        audit.log(pid, "send_dunning", {"channel": channel, "message": msg})
        ok = simulate_dunning(event)
        audit.log(pid, "dunning_result", {"recovered": ok})
        return _result(ok, amount if ok else 0.0,
                       "customer paid after nudge" if ok else "no response to nudge")

    audit.log(pid, "escalate", {"reason": decision.reason})
    return _result(False, 0.0, "escalated / unresolved")
