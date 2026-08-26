"""
Bounded LLM recovery agent.

Given a diagnosed + scored event and a Decision, it executes the recovery
workflow within hard bounds and logs every step to the audit trail. "Bounded"
means it respects stopping rules and compliance - it never runs away.

Messaging uses an LLM when a provider key is configured (see .env.example) and
falls back to a deterministic template otherwise - so the system always sends a
compliant message and still runs fully offline with zero dependencies. The LLM
only ever runs AFTER the compliance gate (compliance.can_message) has allowed
contact; its output is fenced by a system prompt, length-capped, and any failure
(no key, timeout, malformed response) degrades to the template.
"""

import os

# Load .env (if present) so LLM keys/config reach os.getenv. python-dotenv is optional:
# if it isn't installed (e.g. the zero-dependency demo), we skip it and use templates.
# We target the repo-root .env explicitly so it loads regardless of the current dir.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))
except Exception:
    pass

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

# The LLM writes the message body, but it is fenced by this instruction so it can
# never invent an offer, a deadline, or account details - the compliance-critical part.
_LLM_SYSTEM = (
    "You write a single short payment-recovery message for an Indian fintech customer. "
    "Hard rules: one or two sentences, under 300 characters, plain text, no emojis. "
    "State the amount in rupees and the reason the payment failed, then a clear next step. "
    "Never invent discounts, refunds, deadlines, links, or account/card details. "
    "Never promise anything. Address the customer by the name provided."
)

# Human-readable reason per failure code, handed to the model as grounding.
_REASONS = {
    "insufficient_funds": "insufficient balance",
    "expired_card": "the saved card has expired",
    "do_not_honor": "the bank declined the charge",
    "3ds_failure": "the 3-D Secure authentication step failed",
    "bank_downtime": "a temporary bank outage",
    "network_error": "a transient network error",
}


def _template_message(event: dict) -> str:
    """Deterministic fallback - always compliant, always offline."""
    tmpl = TEMPLATES.get(event["failure_code"], TEMPLATES["default"])
    return tmpl.format(name=event.get("customer_id", "there"), amount=round(float(event["amount"])))


def _llm_message(event: dict):
    """
    Provider-agnostic LLM slot-filling via REST. Returns a clean one-line string, or
    None on ANY problem (no key configured, network/timeout, malformed or overlong
    response) so the caller falls back to the template. `httpx` is imported lazily so
    the zero-dependency demo (no key set) never needs it installed.
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider:  # auto-detect from whichever key is present
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            return None  # nothing configured -> stay fully offline

    reason = _REASONS.get(event["failure_code"], "a payment failure")
    user = (f"Customer name: {event.get('customer_id', 'there')}. "
            f"Amount: Rs.{round(float(event['amount']))}. "
            f"Failure reason: {reason} (code: {event['failure_code']}). "
            f"Write the recovery message.")

    try:
        import httpx  # lazy: only imported when a provider key is actually configured
        model = (os.getenv("LLM_MODEL") or "").strip()

        if provider == "openai":
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={"model": model or "gpt-4o-mini",
                      "messages": [{"role": "system", "content": _LLM_SYSTEM},
                                   {"role": "user", "content": user}],
                      "temperature": 0.4, "max_tokens": 120},
                timeout=8.0)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]

        elif provider == "anthropic":
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                         "anthropic-version": "2023-06-01"},
                json={"model": model or "claude-3-5-haiku-latest", "max_tokens": 120,
                      "system": _LLM_SYSTEM,
                      "messages": [{"role": "user", "content": user}]},
                timeout=8.0)
            resp.raise_for_status()
            text = resp.json()["content"][0]["text"]

        elif provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model or 'gemini-1.5-flash'}:generateContent?key={key}",
                json={"system_instruction": {"parts": [{"text": _LLM_SYSTEM}]},
                      "contents": [{"parts": [{"text": user}]}],
                      "generationConfig": {"temperature": 0.4, "maxOutputTokens": 120}},
                timeout=8.0)
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        else:
            return None  # unknown provider name -> template
    except Exception:
        return None  # never let a messaging failure break the recovery workflow

    text = " ".join((text or "").split())   # collapse newlines/whitespace to one line
    if not text or len(text) > 320:          # guardrail: empty or overlong -> fall back
        return None
    return text


def generate_message(event: dict) -> str:
    """
    Dunning message for one event. Uses the configured LLM when available, otherwise a
    deterministic template. Either way the return is a single compliant line: the LLM
    path is fenced by a system prompt + length cap and degrades to the template on any
    failure, so behaviour is safe and fully offline-capable.
    """
    return _llm_message(event) or _template_message(event)


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
