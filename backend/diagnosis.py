"""
Root-cause diagnosis: map a raw failure code to a human root cause + a hint at
the right recovery action. This is the "diagnose" stage the Track 03 brief asks
for. Start rules-based (clear and explainable); optionally let the LLM phrase the
customer-facing explanation later.
"""

# failure_code -> (human cause, suggested recovery action)
ROOT_CAUSE = {
    "insufficient_funds": ("Insufficient funds", "retry_later_morning"),
    "do_not_honor":       ("Issuer declined (do-not-honor)", "retry_or_switch_method"),
    "bank_downtime":      ("Bank temporarily down", "retry_after_downtime"),
    "expired_card":       ("Card expired", "request_new_method"),  # blind retry won't help
    "network_error":      ("Transient network error", "retry_soon"),
    "3ds_failure":        ("Authentication (3DS) dropped", "retry_with_auth_nudge"),
    "other":              ("Unclassified failure", "retry_conservative"),
}


def diagnose(failure_code: str) -> dict:
    cause, suggested_action = ROOT_CAUSE.get(failure_code, ROOT_CAUSE["other"])
    return {
        "failure_code": failure_code,
        "cause": cause,
        "suggested_action": suggested_action,
    }
