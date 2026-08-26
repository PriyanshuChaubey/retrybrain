"""
Train the retry-success model on data/history.csv.

Pipeline: OneHotEncoder(categoricals) + passthrough(numerics) -> classifier.
We train a LogisticRegression baseline and a GradientBoosting model, report AUC
for both, and save the better pipeline to retry_model.pkl.

Run on your machine (needs `pip install -r requirements.txt`):
    python -m backend.model.train        # preferred
    # or: python backend/model/train.py  (sys.path shim below makes this work too)
"""

import os
import sys

# Make `import backend...` work whether run as a module or a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from backend.model.features import CATEGORICAL, NUMERIC, FEATURES, TARGET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
HISTORY = os.path.join(REPO, "data", "history.csv")
MODEL_PATH = os.path.join(HERE, "retry_model.pkl")


def load():
    df = pd.read_csv(HISTORY)
    # CSV stores booleans as text; coerce to 0/1.
    df["in_downtime_window"] = (
        df["in_downtime_window"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    )
    return df


def build_pipeline(classifier):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),   # scale numerics so logreg converges (amount ~5000 vs rate ~1)
    ])
    return Pipeline([("pre", pre), ("clf", classifier)])


def main():
    df = load()
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    candidates = {
        "logreg": LogisticRegression(max_iter=2000),
        "gboost": GradientBoostingClassifier(random_state=42),
    }

    best_name, best_auc, best_model = None, -1.0, None
    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        pipe.fit(Xtr, ytr)
        auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])
        print(f"{name:8s} AUC = {auc:.3f}")
        if auc > best_auc:
            best_name, best_auc, best_model = name, auc, pipe

    print(f"\nBest: {best_name} (AUC={best_auc:.3f})")
    print(classification_report(yte, best_model.predict(Xte), digits=3))
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
