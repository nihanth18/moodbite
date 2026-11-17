"""Ranking: rule-based mood/craving -> feature weights, plus a LinUCB contextual bandit.

Score = rule score (rating, distance, mood rules, cravings, intent match)
        + bandit.weight * LinUCB(mood ⊗ restaurant, time of day)   once there's enough feedback.
The top k are then picked greedily with a per-cuisine cap so the list isn't all one cuisine.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from emotion import EMOTIONS, MoodVector
from places import TAGS, Restaurant

DEFAULT_RULES_PATH = Path(__file__).parent / "config" / "mood_rules.yaml"

MOOD_FEATURES = ("bias", "valence", "arousal", "stress", "sad", "happy")
REST_FEATURES = ("bias", "rating", "closeness", "intent_match", *TAGS)
# Candidates are already 4.0+, so spread 3.5..5.0 over 0..1 to make rating differences count.
RATING_FLOOR, RATING_CEIL = 3.5, 5.0
TIME_BUCKETS = ("morning", "afternoon", "evening", "night")
CONTEXT_DIM = len(MOOD_FEATURES) * len(REST_FEATURES) + len(TIME_BUCKETS)
FEATURE_NAMES = set(REST_FEATURES) - {"bias"}


def load_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for rule in cfg.get("rules", []):
        for key in rule.get("when", {}):
            _parse_condition(key)  # fail fast on typos
        _check_weights(rule.get("weights", {}), rule.get("name", "?"))
    for name, craving in (cfg.get("cravings") or {}).items():
        _check_weights(craving.get("weights", {}), name)
    return cfg


def _check_weights(weights: dict[str, float], owner: str) -> None:
    bad = set(weights) - FEATURE_NAMES
    if bad:
        raise ValueError(f"unknown feature(s) {sorted(bad)} in {owner!r}; "
                         f"use {sorted(FEATURE_NAMES)}")


@dataclass
class Ranked:
    restaurant: Restaurant
    score: float
    why: str
    policy: str
    features: dict[str, Any]


# --- shared feature helpers ----------------------------------------------------------------
def mood_signals(mood: MoodVector) -> dict[str, float]:
    return {**mood.probs, "valence": mood.valence, "arousal": mood.arousal, "stress": mood.stress}


def feature_value(features: dict[str, Any], name: str) -> float:
    if name == "rating":
        # Prefer the review-count-adjusted rating; fall back to the raw one (older log rows).
        r = features.get("weighted_rating")
        r = features.get("rating") if r is None else r
        if r is None:
            return 0.0
        return float(np.clip((r - RATING_FLOOR) / (RATING_CEIL - RATING_FLOOR), 0, 1))
    if name == "bias":
        return 1.0
    return float(features.get(name, 0.0) or 0.0)


def _parse_condition(key: str) -> tuple[str, str]:
    signal, _, op = key.rpartition("_")
    valid = set(EMOTIONS) | {"valence", "arousal", "stress"}
    if op not in ("above", "below") or signal not in valid:
        raise ValueError(f"bad rule condition {key!r}; use <signal>_above|_below with signal "
                         f"in {sorted(valid)}")
    return signal, op


def rule_matches(when: dict[str, float], signals: dict[str, float]) -> bool:
    for key, threshold in when.items():
        signal, op = _parse_condition(key)
        v = signals[signal]
        if op == "above" and not v > threshold:
            return False
        if op == "below" and not v < threshold:
            return False
    return True


def active_rules(cfg: dict[str, Any], mood: MoodVector) -> list[dict[str, Any]]:
    s = mood_signals(mood)
    return [r for r in cfg.get("rules", []) if rule_matches(r.get("when", {}), s)]


def active_cravings(cfg: dict[str, Any], cravings: Iterable[str]) -> list[dict[str, Any]]:
    table = cfg.get("cravings") or {}
    return [{"name": c, **table[c]} for c in cravings if c in table]


def search_queries(cfg: dict[str, Any], mood: MoodVector,
                   cravings: Iterable[str] = ()) -> list[str]:
    """Extra Google searches for this mood: cravings first (stated directly), then mood rules."""
    limit = int((cfg.get("retrieval") or {}).get("max_queries", 3))
    queries: list[str] = []
    for item in active_cravings(cfg, cravings) + active_rules(cfg, mood):
        for q in item.get("queries", []):
            if q not in queries:
                queries.append(q)
    return queries[:limit]


def _contribution(weights: dict[str, float], features: dict[str, Any]) -> float:
    return sum(w * feature_value(features, n) for n, w in weights.items())


def explain(cfg: dict[str, Any], rules: list[dict[str, Any]], features: dict[str, Any],
            mood_weight: float = 1.0, cravings: list[dict[str, Any]] | None = None) -> str:
    """One-line 'why': the craving or mood rule that contributes most to this place's score.

    The matching tags and the distance are shown separately on the card.
    """
    best, best_contrib = None, 0.0
    for craving in cravings or []:
        c = _contribution(craving.get("weights", {}), features)
        if c > best_contrib:
            best, best_contrib = craving, c
    for rule in rules:
        c = _contribution(rule.get("weights", {}), features) * mood_weight
        if c > best_contrib:
            best, best_contrib = rule, c
    if best is None:
        return f"{cfg.get('default_why', 'A top-rated place near you')}."
    return f"{best['why']}."


# --- rules ---------------------------------------------------------------------------------
class RuleRanker:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def score(self, mood: MoodVector, features: dict[str, Any],
              rules: list[dict[str, Any]] | None = None, mood_weight: float = 1.0,
              cravings: list[dict[str, Any]] | None = None) -> float:
        rules = active_rules(self.cfg, mood) if rules is None else rules
        total = _contribution(self.cfg.get("base_weights", {}), features)
        for rule in rules:
            total += mood_weight * _contribution(rule.get("weights", {}), features)
        for craving in cravings or []:
            total += _contribution(craving.get("weights", {}), features)
        return float(total)


# --- LinUCB --------------------------------------------------------------------------------
def time_bucket(hour: int) -> str:
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def mood_vector(mood: MoodVector) -> np.ndarray:
    s = mood_signals(mood)
    return np.array([1.0] + [float(s[n]) for n in MOOD_FEATURES[1:]])


def restaurant_vector(features: dict[str, Any]) -> np.ndarray:
    return np.array([feature_value(features, n) for n in REST_FEATURES])


def context_vector(mood: MoodVector, features: dict[str, Any], hour: int) -> np.ndarray:
    """[mood ⊗ restaurant features, one-hot time_of_day]"""
    tod = np.zeros(len(TIME_BUCKETS))
    tod[TIME_BUCKETS.index(time_bucket(hour))] = 1.0
    return np.concatenate([np.outer(mood_vector(mood), restaurant_vector(features)).ravel(), tod])


class LinUCB:
    """Single shared-parameter LinUCB: reward ≈ θ·x, score = θ·x + α·sqrt(xᵀA⁻¹x)."""

    def __init__(self, dim: int = CONTEXT_DIM, alpha: float = 0.6, ridge: float = 1.0):
        self.dim, self.alpha = dim, alpha
        self.A = ridge * np.eye(dim)
        self.b = np.zeros(dim)
        self.n_updates = 0
        self._A_inv: np.ndarray | None = None

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n_updates += 1
        self._A_inv = None

    @property
    def A_inv(self) -> np.ndarray:
        if self._A_inv is None:
            self._A_inv = np.linalg.inv(self.A)
        return self._A_inv

    @property
    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def predict(self, x: np.ndarray) -> float:
        return float(self.theta @ x)

    def ucb(self, x: np.ndarray) -> float:
        return float(self.theta @ x + self.alpha * np.sqrt(max(x @ self.A_inv @ x, 0.0)))

    def fit(self, events: list[dict[str, Any]]) -> "LinUCB":
        for ev in events:
            x = context_vector(MoodVector.from_dict(ev["mood"]), ev["restaurant"], ev["hour"])
            self.update(x, float(ev["reward"]))
        return self


# --- diversity -----------------------------------------------------------------------------
def diversify(ranked: list[Ranked], k: int, max_per_cuisine: int = 2,
              repeat_penalty: float = 0.4) -> list[Ranked]:
    """Greedy top-k: each earlier pick of the same cuisine costs `repeat_penalty`, and no
    cuisine appears more than `max_per_cuisine` times (unknown cuisines aren't capped)."""
    pool = sorted(ranked, key=lambda x: x.score, reverse=True)
    picked: list[Ranked] = []
    counts: Counter[str] = Counter()
    while pool and len(picked) < k:
        best_i, best_s = None, -np.inf
        for i, item in enumerate(pool):
            c = item.restaurant.cuisine
            known = c and c != "unknown"
            if known and counts[c] >= max_per_cuisine:
                continue
            s = item.score - (repeat_penalty * counts[c] if known else 0.0)
            if s > best_s:
                best_i, best_s = i, s
        if best_i is None:  # everything left is capped; fill with the best remaining
            best_i = 0
        item = pool.pop(best_i)
        counts[item.restaurant.cuisine] += 1
        picked.append(item)
    return picked


# --- combined recommender ------------------------------------------------------------------
class Recommender:
    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or load_rules()
        bandit = self.cfg.get("bandit", {})
        self.min_events = int(bandit.get("min_events", 20))
        self.alpha = float(bandit.get("alpha", 0.6))
        self.ridge = float(bandit.get("ridge", 1.0))
        self.bandit_weight = float(bandit.get("weight", 1.0))
        div = self.cfg.get("diversity") or {}
        self.max_per_cuisine = int(div.get("max_per_cuisine", 2))
        self.repeat_penalty = float(div.get("repeat_penalty", 0.4))
        self.rules = RuleRanker(self.cfg)
        self.bandit: LinUCB | None = None

    def retrain(self, events: list[dict[str, Any]]) -> LinUCB | None:
        """Refit the bandit from logged events; None if there isn't enough feedback yet."""
        if len(events) < self.min_events:
            self.bandit = None
        else:
            self.bandit = LinUCB(alpha=self.alpha, ridge=self.ridge).fit(events)
        return self.bandit

    @property
    def policy(self) -> str:
        return "linucb" if self.bandit is not None else "rules"

    def queries(self, mood: MoodVector, cravings: Iterable[str] = ()) -> list[str]:
        return search_queries(self.cfg, mood, cravings)

    def rank(self, mood: MoodVector, restaurants: list[Restaurant], max_distance_m: float,
             hour: int, k: int = 5, mood_weight: float = 1.0,
             cravings: Iterable[str] = ()) -> list[Ranked]:
        rules = active_rules(self.cfg, mood)
        crav = active_cravings(self.cfg, cravings)
        ranked = []
        for r in restaurants:
            feats = r.features(max_distance_m)
            why = explain(self.cfg, rules, feats, mood_weight, crav)
            score = self.rules.score(mood, feats, rules, mood_weight, crav)
            if self.bandit is not None:
                score += self.bandit_weight * self.bandit.ucb(context_vector(mood, feats, hour))
                why = why.rstrip(".") + ", tuned by your past picks."
            ranked.append(Ranked(r, score, why, self.policy, feats))
        return diversify(ranked, k, self.max_per_cuisine, self.repeat_penalty)
