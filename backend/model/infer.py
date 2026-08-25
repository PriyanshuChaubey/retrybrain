"""
Score a single event -> P(retry succeeds).

If a trained model (retry_model.pkl) exists, use it. Otherwise fall back to a
domain-prior HEURISTIC so the whole pipeline runs before you've trained (and in
environments without scikit-learn). Train the model on your machine to replace
the heuristic automatically - no other code changes needed.
"""

import os

from backend.model.features import FEATURES, normalize

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "retry_model.pkl")

# Rough prior belief that a *well-handled* recovery works, by failure type.
# (This is a stand-in for the ML model, not the ground truth in simulator.py.)
HEURISTIC_PRIOR = {
    "network_error": 0.75,
    "insufficient_funds": 0.55,
    "bank_downtime": 0.55,     # believes a retry-after-window will work
    "3ds_failure": 0.45,
    "other": 0.35,
    "do_not_honor": 0.30,
    "expired_card": 0.05,      # a retry won't help -> policy should dun/switch
}

_model = None


def load_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        import joblib  # lazy: only needed when a model file exists
        _model = joblib.load(MODEL_PATH)
    return _model


def score(event: dict) -> float:
    model = load_model()
    if model is not None:
        import pandas as pd  # lazy import
        row = pd.DataFrame([normalize(event)])[FEATURES]
        return float(model.predict_proba(row)[0, 1])

    # --- heuristic fallback ---
    p = HEURISTIC_PRIOR.get(event["failure_code"], 0.35)
    p *= 0.9 ** (int(event.get("attempt_number", 1)) - 1)   # diminishing returns
    return max(0.0, min(1.0, p))
