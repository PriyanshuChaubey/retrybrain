"""
Tests for the compliance guardrails - the "compliant escalation" Track 03 asks
for: DND blocks contact, quiet hours are respected, the escalation ladder is
walked in order and terminates.
"""

from backend.agent import compliance


def test_dnd_blocks_contact():
    allowed, _ = compliance.can_message({"dnd": True})
    assert allowed is False
    allowed, _ = compliance.can_message({"dnd": False})
    assert allowed is True


def test_quiet_hours_window():
    # No customer messaging 21:00 (inclusive) through 08:00 (exclusive).
    assert compliance.in_quiet_hours(22) is True
    assert compliance.in_quiet_hours(3) is True
    assert compliance.in_quiet_hours(21) is True
    assert compliance.in_quiet_hours(7) is True
    assert compliance.in_quiet_hours(8) is False
    assert compliance.in_quiet_hours(12) is False


def test_escalation_ladder_is_ordered_and_terminates():
    assert compliance.next_escalation(None) == "reminder"
    assert compliance.next_escalation("reminder") == "alternate_method"
    assert compliance.next_escalation("alternate_method") == "final_notice"
    assert compliance.next_escalation("final_notice") == "human_handoff"
    assert compliance.next_escalation("human_handoff") is None
