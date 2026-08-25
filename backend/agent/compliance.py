"""
Compliance guardrails for customer contact - the "compliant escalation" the
Track 03 brief asks for. Respect consent/DND, quiet hours, and a defined
escalation ladder. Keeping these here (not scattered in the agent) makes them
easy to audit and defend.
"""

QUIET_START, QUIET_END = 21, 8   # no customer messaging 9pm-8am local
ESCALATION_LADDER = ["reminder", "alternate_method", "final_notice", "human_handoff"]


def can_message(event: dict) -> tuple[bool, str]:
    """Whether we're allowed to contact this customer. Returns (allowed, reason)."""
    if event.get("dnd"):
        return False, "customer on DND / opted out of comms"
    return True, "ok"


def in_quiet_hours(hour: int) -> bool:
    """True if `hour` falls in the do-not-disturb window (messages get held)."""
    return hour >= QUIET_START or hour < QUIET_END


def next_escalation(current: str | None) -> str | None:
    """Advance the escalation ladder; None means the ladder is exhausted."""
    if current is None:
        return ESCALATION_LADDER[0]
    i = ESCALATION_LADDER.index(current)
    return ESCALATION_LADDER[i + 1] if i + 1 < len(ESCALATION_LADDER) else None
