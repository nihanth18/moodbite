"""Offline baseline: logistic regression (SGD) on logged feedback.

    python train.py                 # train on moodbite.db, print accuracy / AUC
    python train.py --simulate 200  # first add 200 synthetic events (for trying things out)
"""
from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from emotion import EMOTIONS, MoodVector
from places import demo_restaurants
from ranker import context_vector
from storage import Storage

MIN_EVENTS = 10


def build_dataset(events: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([context_vector(MoodVector.from_dict(e["mood"]), e["restaurant"], e["hour"])
                  for e in events])
    y = np.array([int(e["reward"]) for e in events])
    return X, y


def train_baseline(events: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Fit SGD logistic regression; returns metrics dict (or a 'skipped' reason)."""
    n = len(events)
    if n < MIN_EVENTS:
        return {"n_events": n, "skipped": f"need at least {MIN_EVENTS} events, have {n}"}
    X, y = build_dataset(events)
    if len(set(y)) < 2:
        return {"n_events": n, "skipped": "need both 'Going here' and 'Not feeling it' feedback"}
    stratify = y if min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed,
                                              stratify=stratify)
    clf = SGDClassifier(loss="log_loss", alpha=1e-3, max_iter=2000, tol=1e-4,
                        random_state=seed)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    majority = int(np.bincount(y_tr).argmax())
    metrics = {
        "n_events": n,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "positive_rate": round(float(y.mean()), 3),
        "accuracy": round(float(accuracy_score(y_te, pred)), 3),
        "majority_baseline_accuracy": round(float((y_te == majority).mean()), 3),
        "auc": round(float(roc_auc_score(y_te, proba)), 3) if len(set(y_te)) > 1 else None,
    }
    return metrics


def simulate_events(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Synthetic user: tired -> comfort/quick, happy+energetic -> lively, stressed -> dessert."""
    rng = random.Random(seed)
    places = demo_restaurants()
    start = datetime(2026, 1, 1, 8)
    events = []
    for i in range(n):
        raw = [rng.random() ** 2 for _ in EMOTIONS]
        top = rng.randrange(len(EMOTIONS))
        raw[top] += 2.0
        s = sum(raw)
        probs = {e: v / s for e, v in zip(EMOTIONS, raw)}
        valence = float(np.clip(probs["happy"] - probs["sad"] - probs["angry"] + rng.gauss(0, .2),
                                -1, 1))
        arousal = float(np.clip(probs["surprise"] + probs["angry"] + probs["fear"]
                                - probs["sad"] - probs["neutral"] * .5 + rng.gauss(0, .2), -1, 1))
        mood = MoodVector(probs, valence, arousal)
        r = rng.choice(places)
        f = r.features(2000)
        u = 0.4 * f["closeness"] - 0.3
        u += 1.5 * (arousal < -0.1) * (f["comfort_food"] + f["quick_service"])
        u += 1.5 * (valence > 0.3 and arousal > 0.2) * f["lively"]
        u += 1.5 * (mood.stress > 0.35) * f["dessert"]
        u -= 1.0 * (arousal < -0.1) * f["lively"]
        reward = rng.random() < 1 / (1 + np.exp(-2 * u))
        ts = start + timedelta(hours=7 * i)
        events.append({"mood": mood.to_dict(), "restaurant": f, "hour": ts.hour,
                       "reward": int(reward), "ts": ts})
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--db", default=os.getenv("MOODBITE_DB", "moodbite.db"))
    ap.add_argument("--simulate", type=int, default=0,
                    help="insert N synthetic feedback events before training")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    store = Storage(args.db)
    if args.simulate:
        for e in simulate_events(args.simulate, args.seed):
            store.log_feedback(e["mood"], e["restaurant"],
                               "going" if e["reward"] else "not_feeling_it", "simulated", e["ts"])
        print(f"Inserted {args.simulate} synthetic events into {args.db}")

    metrics = train_baseline(store.load_events(), seed=args.seed)
    print("Offline baseline: SGD logistic regression")
    for k, v in metrics.items():
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
