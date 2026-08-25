"""
Tests for the decision engine - the stopping rules and intervention choices that
Track 03 cares about. Run:  pytest -q
"""

from backend.decision_engine import decide, optimal_retry, MAX_ATTEMPTS


def _ev(**kw):
    base = {"failure_code": "network_error", "attempt_number": 1,
            "timestamp": "2026-08-25T18:00:00", "in_downtime_window": False}
    base.update(kw)
    return base


def test_stops_after_max_attempts():
    # An event on its MAX_ATTEMPTS-th try must stop, even with a high score.
    assert decide(_ev(attempt_number=MAX_ATTEMPTS), 0.99).action == "stop"


def test_expired_card_switches_method_not_retry():
    # A blind retry never recovers an expired card -> switch_method regardless of score.
    assert decide(_ev(failure_code="expired_card"), 0.99).action == "switch_method"


def test_high_probability_triggers_retry():
    d = decide(_ev(failure_code="network_error"), 0.9)
    assert d.action == "retry"
    assert d.at_time is not None and d.retry_hour is not None


def test_low_probability_duns_instead_of_retrying():
    assert decide(_ev(failure_code="other"), 0.05).action == "dun"


def test_optimal_retry_timing_rules():
    # insufficient_funds waits for the morning; bank_downtime waits out the window.
    _, hour, dt_over = optimal_retry(_ev(failure_code="insufficient_funds"))
    assert hour == 9 and dt_over is False
    _, _, dt_over_bank = optimal_retry(_ev(failure_code="bank_downtime", in_downtime_window=True))
    assert dt_over_bank is True
