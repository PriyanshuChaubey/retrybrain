"""
Smoke test for the batch runner - the core claim of the project: across the whole
synthetic batch, RetryBrain recovers at least as much as a naive immediate-retry
baseline, recovers real money, and produces an (honest) exception list. Seeded, so
this is deterministic.
"""

from backend.runner import run_batch


def test_batch_runs_and_beats_baseline():
    m = run_batch()

    assert m["total_payments"] >= 50                       # Track 03: 50+ record batch
    assert m["rupees_recovered"] > 0
    assert m["recovered_count"] >= m["baseline"]["recovered_count"]   # the core claim
    assert m["by_failure_code"]                            # per-cause breakdown exists
    assert isinstance(m["exceptions"], list)               # honest exception list
    assert m["recovered_count"] + m["exception_count"] == m["total_payments"]


def test_reproducible():
    # Same seed -> identical headline number on a re-run.
    assert run_batch()["recovered_count"] == run_batch()["recovered_count"]
