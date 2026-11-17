import numpy as np
import pytest

from conftest import make_mood
from places import demo_restaurants, filter_candidates
from ranker import (
    CONTEXT_DIM, LinUCB, Recommender, RuleRanker, active_rules, context_vector, feature_value,
    load_rules, rule_matches, time_bucket,
)
from train import simulate_events

MAX_M = 2000


@pytest.fixture(scope="module")
def cfg():
    return load_rules()


@pytest.fixture(scope="module")
def candidates():
    return filter_candidates(demo_restaurants(), max_distance_m=MAX_M)


TIRED = dict(valence=-0.1, arousal=-0.6, neutral=0.6, sad=0.2, happy=0.2)
UPBEAT = dict(valence=0.7, arousal=0.6, happy=0.85, surprise=0.15)
STRESSED = dict(valence=-0.4, arousal=0.3, angry=0.3, fear=0.25, neutral=0.45)


# --- rules -----------------------------------------------------------------------------------
def test_config_loads_and_validates(cfg, tmp_path):
    assert cfg["bandit"]["min_events"] == 20
    bad = tmp_path / "bad.yaml"
    bad.write_text("rules:\n  - name: x\n    when: {hapy_above: 0.5}\n    weights: {}\n")
    with pytest.raises(ValueError, match="hapy_above"):
        load_rules(bad)


def test_rule_matches():
    s = {"valence": 0.5, "arousal": -0.3}
    assert rule_matches({"valence_above": 0.3, "arousal_below": 0.0}, s)
    assert not rule_matches({"valence_above": 0.6}, s)
    assert rule_matches({}, s)


def test_active_rules(cfg):
    assert [r["name"] for r in active_rules(cfg, make_mood(**TIRED))] == ["low_energy"]
    assert "upbeat" in [r["name"] for r in active_rules(cfg, make_mood(**UPBEAT))]
    assert "stressed" in [r["name"] for r in active_rules(cfg, make_mood(**STRESSED))]
    assert active_rules(cfg, make_mood(neutral=1.0)) == []


def test_feature_value_defaults():
    assert feature_value({"rating": None}, "rating") == 0.0
    assert feature_value({"rating": 5.0}, "rating") == 1.0
    assert feature_value({"rating": 3.0}, "rating") == 0.0
    # weighted rating takes precedence over the raw one
    assert feature_value({"rating": 4.9, "weighted_rating": 4.25}, "rating") == 0.5
    assert feature_value({}, "dessert") == 0.0


def test_rating_dominates_when_mood_is_neutral(cfg, candidates):
    top = Recommender(cfg).rank(make_mood(neutral=1.0), candidates, MAX_M, hour=12, k=3)
    best = sorted(candidates, key=lambda r: r.weighted_rating, reverse=True)[:5]
    assert {t.restaurant.id for t in top} <= {r.id for r in best}


def test_mood_changes_the_score(cfg):
    rr = RuleRanker(cfg)
    comfort = {"rating": 4.0, "closeness": 0.5, "comfort_food": 1}
    lively = {"rating": 4.0, "closeness": 0.5, "lively": 1}
    tired, upbeat = make_mood(**TIRED), make_mood(**UPBEAT)
    assert rr.score(tired, comfort) > rr.score(tired, lively)
    assert rr.score(upbeat, lively) > rr.score(upbeat, comfort)


def tags_of(results, tag):
    return [r.features[tag] for r in results]


def test_tired_mood_prefers_comfort_or_quick(cfg, candidates):
    top = Recommender(cfg).rank(make_mood(**TIRED), candidates, MAX_M, hour=20, k=3)
    assert all(r.features["comfort_food"] or r.features["quick_service"] for r in top)
    assert not any(tags_of(top, "lively"))


def test_upbeat_mood_prefers_lively(cfg, candidates):
    top = Recommender(cfg).rank(make_mood(**UPBEAT), candidates, MAX_M, hour=20, k=2)
    assert all(tags_of(top, "lively"))


def test_stressed_mood_prefers_dessert_or_familiar(cfg, candidates):
    top = Recommender(cfg).rank(make_mood(**STRESSED), candidates, MAX_M, hour=20, k=1)[0]
    assert top.features["dessert"] or top.features["familiar"]


def test_top5_and_why(cfg, candidates):
    top = Recommender(cfg).rank(make_mood(**TIRED), candidates, MAX_M, hour=12)
    assert len(top) == 5
    assert all(r.policy == "rules" for r in top)
    assert all("low on energy" in r.why for r in top if r.features["comfort_food"])
    assert all(r.why.endswith(".") and "away" not in r.why for r in top)  # distance is on the card
    scores = [r.score for r in top]
    assert scores == sorted(scores, reverse=True)


def test_neutral_mood_uses_default_why(cfg, candidates):
    top = Recommender(cfg).rank(make_mood(neutral=1.0), candidates, MAX_M, hour=12, k=1)
    assert top[0].why.startswith(cfg["default_why"])


def test_mood_weight_zero_reduces_to_base(cfg, candidates):
    rr = RuleRanker(cfg)
    f = candidates[0].features(MAX_M)
    assert rr.score(make_mood(**TIRED), f, mood_weight=0) == \
        rr.score(make_mood(neutral=1.0), f)


def test_empty_candidates(cfg):
    assert Recommender(cfg).rank(make_mood(**TIRED), [], MAX_M, hour=12) == []


# --- LinUCB ----------------------------------------------------------------------------------
def test_time_bucket():
    assert [time_bucket(h) for h in (6, 13, 19, 23, 2)] == \
        ["morning", "afternoon", "evening", "night", "night"]


def test_context_vector_shape(candidates):
    x = context_vector(make_mood(**TIRED), candidates[0].features(MAX_M), hour=13)
    assert x.shape == (CONTEXT_DIM,)
    assert x[-4:].tolist() == [0, 1, 0, 0]


def test_linucb_learns_reward():
    rng = np.random.default_rng(0)
    bandit = LinUCB(dim=5, alpha=0.1)
    true_theta = np.array([1.0, -1.0, 0.5, 0.0, 0.0])
    for _ in range(500):
        x = rng.normal(size=5)
        bandit.update(x, float(true_theta @ x + rng.normal(0, 0.1)))
    assert np.allclose(bandit.theta, true_theta, atol=0.1)


def test_linucb_exploration_bonus_shrinks():
    b = LinUCB(dim=3, alpha=1.0)
    x = np.array([1.0, 0.0, 0.0])
    before = b.ucb(x)
    for _ in range(50):
        b.update(x, 0.0)
    assert b.ucb(x) < before


def test_fallback_to_rules_below_min_events(cfg):
    rec = Recommender(cfg)
    assert rec.retrain(simulate_events(19)) is None
    assert rec.policy == "rules"
    assert rec.retrain(simulate_events(20)) is not None
    assert rec.policy == "linucb"
    assert rec.bandit.n_updates == 20


def test_bandit_ranking_uses_linucb(cfg, candidates):
    rec = Recommender(cfg)
    rec.retrain(simulate_events(300, seed=1))
    top = rec.rank(make_mood(**TIRED), candidates, MAX_M, hour=20)
    assert len(top) == 5 and all(r.policy == "linucb" for r in top)
    assert all("past picks" in r.why for r in top)


def test_bandit_learns_synthetic_preference(cfg, candidates):
    """The simulated user likes comfort/quick food when tired and dislikes lively places."""
    rec = Recommender(cfg)
    rec.retrain(simulate_events(1500, seed=2))
    rec.bandit.alpha = 0.0  # pure exploitation for the check
    tired = make_mood(**TIRED)
    scored = {r.restaurant.name: r for r in rec.rank(tired, candidates, MAX_M, 20, k=100)}
    comfort = [s.score for s in scored.values()
               if s.features["comfort_food"] or s.features["quick_service"]]
    lively = [s.score for s in scored.values() if s.features["lively"]]
    assert np.mean(comfort) > np.mean(lively)
