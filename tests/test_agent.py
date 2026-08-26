"""
Tests for the recovery agent's messaging layer.

The safety-critical guarantee: with no LLM provider configured, generate_message
falls back to a deterministic, compliant template and the whole system runs fully
offline (the httpx import is lazy, so it is never even reached without a key). When
the LLM path does produce text, that text is used. No network is touched in these
tests.
"""

from backend.agent import agent


def _event(code="insufficient_funds", amount=1499, name="cust_42"):
    return {"payment_id": "p1", "failure_code": code, "amount": amount, "customer_id": name}


def _clear_llm_env(monkeypatch):
    for var in ("LLM_PROVIDER", "LLM_MODEL", "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_offline_fallback_when_no_provider(monkeypatch):
    # With nothing configured, we must get the deterministic template - and it must
    # state the amount. This is the guarantee that keeps the demo zero-dependency.
    _clear_llm_env(monkeypatch)
    msg = agent.generate_message(_event(amount=1499))
    assert isinstance(msg, str) and msg
    assert "1499" in msg
    assert msg == agent._template_message(_event(amount=1499))


def test_llm_message_is_none_without_key(monkeypatch):
    # The provider selector returns None (no lazy httpx import happens) when unconfigured.
    _clear_llm_env(monkeypatch)
    assert agent._llm_message(_event()) is None


def test_template_covers_every_failure_code():
    # Every known code (plus an unknown one) yields a non-empty, single-line message.
    for code in ("insufficient_funds", "expired_card", "do_not_honor",
                 "3ds_failure", "bank_downtime", "network_error", "mystery_code"):
        msg = agent._template_message(_event(code=code))
        assert msg and "\n" not in msg


def test_llm_output_is_used_when_available(monkeypatch):
    # If the LLM path yields text, generate_message returns it verbatim (not the template).
    monkeypatch.setattr(agent, "_llm_message", lambda e: "Custom LLM line about Rs.1499.")
    assert agent.generate_message(_event()) == "Custom LLM line about Rs.1499."


def test_llm_failure_degrades_to_template(monkeypatch):
    # Any LLM failure surfaces as None from _llm_message -> caller must fall back, not crash.
    monkeypatch.setattr(agent, "_llm_message", lambda e: None)
    assert agent.generate_message(_event()) == agent._template_message(_event())
