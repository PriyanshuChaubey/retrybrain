"""
Economics of recovery — turns the batch into a BUDGET-CONSTRAINED allocation problem.

Why this exists (Track 03 / interview point)
---------------------------------------------
Retrying a failed payment is a near-free gateway call, but *contacting a customer*
(a WhatsApp template, an email, and ultimately a human agent) costs real money. At
Razorpay scale you cannot nudge everyone, so recovery stops being "message all
failures" and becomes portfolio allocation: spend each scarce outreach rupee where
its EXPECTED return is highest.

RetryBrain therefore ranks every failed payment by the expected ROI of contacting
it, funds outreach top-down until the budget is exhausted, and lets the unfunded
low-ROI failures fall through to the free retry only — then lists them honestly as
exceptions. Free retries are never gated; only paid outreach is.

Every number in this module is an explicit, documented ASSUMPTION (the levers a
finance/ops team would actually negotiate), not a value fit to the simulator. The
allocator deliberately does NOT know the simulator's true dunning-success rate — it
ranks using domain priors, so it is being tested, not handed the answer.
"""

# --- Per-action cost in rupees (documented assumptions) ----------------------
# Retries are a near-free re-attempt; a WhatsApp Business template and a
# transactional email carry real per-message fees (Indian pricing ballpark); a
# human handoff is support-agent time. These are the cost levers, stated openly.
ACTION_COST = {
    "retry": 0.02,          # gateway re-attempt — effectively free
    "email": 0.10,          # transactional email nudge
    "whatsapp": 0.35,       # WhatsApp Business template message
    "human_handoff": 0.0,   # a FLAG for human review; costing human time is out of scope
    "stop": 0.0,
    "no_action": 0.0,
}


def contact_channel(event: dict) -> str:
    """The paid channel we would use to reach this customer (consent-driven)."""
    return "whatsapp" if event.get("consent_whatsapp") else "email"


def contact_cost(event: dict) -> float:
    """Rupee cost of ONE paid outreach message to this customer."""
    return ACTION_COST[contact_channel(event)]


# --- Expected value of CONTACTING a customer (no oracle peeking) --------------
# Prior belief that a single compliant nudge converts, by failure cause. Grounded
# in the domain, NOT read from simulator.py: an expired card responds best (the
# nudge asks the customer to re-enter card details — the only real fix); bank/
# 3-DS/do-not-honor are moderate; a transient network error rarely needs a human
# nudge at all (a retry fixes it). These priors are intentionally richer than the
# simulator's flat dunning-success rate, which is the whole point — the allocator
# targets, it does not treat all failures alike.
CONTACT_CONVERSION_PRIOR = {
    "expired_card":       0.45,
    "do_not_honor":       0.30,
    "3ds_failure":        0.30,
    "insufficient_funds": 0.25,
    "bank_downtime":      0.12,
    "network_error":      0.08,
    "other":              0.20,
}


def conversion_prior(event: dict) -> float:
    return CONTACT_CONVERSION_PRIOR.get(event.get("failure_code", "other"), 0.20)


def expected_contact_value(event: dict) -> float:
    """
    Expected rupees recovered by paying to contact this customer, NET of cost:
        E[value] = P(convert | cause) * amount - contact_cost
    Positive means the nudge is worth funding on its own merits.
    """
    return conversion_prior(event) * float(event["amount"]) - contact_cost(event)


def expected_roi(event: dict) -> float:
    """
    Expected rupees recovered per rupee spent contacting (>= 0). This is the
    allocation ranking key: fund the highest-ROI failures first. Uses only
    features known at decision time plus the documented priors.
    """
    cost = contact_cost(event)
    gross = conversion_prior(event) * float(event["amount"])
    return gross / cost if cost > 0 else float("inf")


class Budget:
    """
    A hard outreach budget with ROI-ordered allocation.

    plan() decides UP FRONT which payments are worth contacting: sort candidates by
    expected ROI (highest first) and reserve each one's expected outreach cost until
    the budget runs out. At run time try_spend() enforces the budget as a hard cap
    (a safety net); because we paced by expected cost, the cap rarely binds. Free
    actions (retries) are never routed through the budget.

    An "unlimited" budget funds everything and never gates — this is the default so
    the headline demo shows RetryBrain's full capability, unchanged.
    """

    def __init__(self, limit: float):
        self.limit = float(limit)
        self.spent = 0.0
        self.funded = set()      # payment_ids chosen for paid outreach
        self._planned = False

    @classmethod
    def unlimited(cls) -> "Budget":
        return cls(float("inf"))

    @property
    def is_unlimited(self) -> bool:
        return self.limit == float("inf")

    def plan(self, batch) -> "Budget":
        """Choose the funded set by descending expected ROI within the budget."""
        self._planned = True
        if self.is_unlimited:
            return self
        # A DND customer cannot be legally contacted, so paid outreach is never an
        # option for them — exclude them from funding entirely (the runner still
        # routes them to compliant handling + human handoff). Among the reachable,
        # only fund those where contacting has positive expected net value.
        candidates = [ev for ev in batch
                      if not ev.get("dnd") and expected_contact_value(ev) > 0]
        candidates.sort(key=expected_roi, reverse=True)
        running = 0.0
        for ev in candidates:
            c = contact_cost(ev)
            if running + c <= self.limit:
                self.funded.add(ev["payment_id"])
                running += c
        return self

    def is_funded(self, payment_id: str) -> bool:
        """True if this payment was allocated outreach budget (always True if unlimited)."""
        return self.is_unlimited or payment_id in self.funded

    def try_spend(self, amount: float) -> bool:
        """Hard cap: spend if it fits, else refuse. Near-free retries pass trivially."""
        if self.spent + amount <= self.limit + 1e-9:
            self.spent += amount
            return True
        return False


# --- Costing a completed recovery workflow -----------------------------------
def workflow_cost(retries_used: int, contacts_used: int, channel) -> dict:
    """
    Break a finished payment's spend into retry vs outreach rupees. `channel` is the
    paid channel actually used (None if the payment was never contacted).
    """
    retry_spent = retries_used * ACTION_COST["retry"]
    outreach_spent = contacts_used * ACTION_COST.get(channel, 0.0) if channel else 0.0
    return {
        "channel": channel,
        "retry_spent": round(retry_spent, 4),
        "outreach_spent": round(outreach_spent, 4),
        "cost_spent": round(retry_spent + outreach_spent, 4),
    }
