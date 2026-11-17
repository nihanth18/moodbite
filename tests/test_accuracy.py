"""Tests for the accuracy work: richer tags, mood-driven search, taste filters, diversity,
hybrid bandit scoring."""
import itertools
from unittest.mock import MagicMock

import pytest

from conftest import make_mood
from places import (
    GENERIC_QUERY, NEARBY, PlacesClient, Restaurant, demo_restaurants, demo_search,
    filter_candidates, in_cuisine_group, infer_pure_veg, infer_tags, merge_results,
    parse_place,
)
from ranker import Ranked, Recommender, diversify, explain, load_rules, search_queries
from train import simulate_events

MAX_M = 2000
TIRED = dict(valence=-0.1, arousal=-0.6, neutral=0.6, sad=0.2, happy=0.2)
UPBEAT = dict(valence=0.7, arousal=0.6, happy=0.85, surprise=0.15)
STRESSED = dict(valence=-0.4, arousal=0.3, angry=0.3, fear=0.25, neutral=0.45)
CALM = dict(valence=0.4, arousal=-0.2, happy=0.5, neutral=0.5)


@pytest.fixture(scope="module")
def cfg():
    return load_rules()


# --- richer tags -----------------------------------------------------------------------------
class TestAttributeTags:
    def test_live_music_or_cocktails_make_a_place_lively(self):
        assert infer_tags("Plain Name", ["restaurant"], attrs={"liveMusic": True})["lively"] == 1
        assert infer_tags("Plain", ["restaurant"], attrs={"servesCocktails": True})["lively"] == 1
        assert infer_tags("Plain", ["restaurant"], attrs={"servesBeer": True})["lively"] == 0.5

    def test_serves_dessert_is_only_weak_evidence(self):
        """Most restaurants serve dessert, so it mustn't look like a dessert shop."""
        weak = infer_tags("Plain", ["restaurant"], attrs={"servesDessert": True})["dessert"]
        shop = infer_tags("Plain", ["dessert_shop"])["dessert"]
        assert 0 < weak < shop == 1.0

    def test_outdoor_from_attribute_or_description(self):
        assert infer_tags("X", ["restaurant"], attrs={"outdoorSeating": True})["outdoor"] == 1
        assert infer_tags("X", ["restaurant"], summary="Rooftop dining")["outdoor"] == 1

    def test_description_keywords_count(self):
        t = infer_tags("Hotel Shadab", ["restaurant"],
                       summary="Traditional Hyderabadi biryani & haleem in a homey space.")
        assert t["comfort_food"] == 1 and t["spicy"] == 1

    def test_lively_attribute_clears_quiet(self):
        t = infer_tags("Garden Cafe", ["cafe"], attrs={"liveMusic": True})
        assert t["lively"] == 1 and t["quiet"] == 0

    def test_light_ignores_ambiguous_description_words(self):
        t = infer_tags("Okra", ["european_restaurant"],
                       summary="Indian & global fare served in a modern, light-filled space")
        assert t["light"] == 0

    def test_light_and_coffee(self):
        assert infer_tags("Green Bowl", ["vegan_restaurant"])["light"] == 1
        assert infer_tags("Brew Lab", ["coffee_shop"])["coffee"] == 1


class TestVeg:
    def test_pure_veg_detection(self):
        assert infer_pure_veg("Udupi Pure Veg Tiffins", ["restaurant"])
        assert infer_pure_veg("Leaf", ["vegetarian_restaurant"])
        assert not infer_pure_veg("Veg & Non Veg Biryani", ["restaurant"])
        assert not infer_pure_veg("Chicken Corner", ["restaurant"])
        assert not infer_pure_veg("Grand Hotel", ["indian_restaurant"])

    def test_veg_options_from_attribute(self):
        r = parse_place({"id": "v", "displayName": {"text": "Grand Hotel"},
                         "servesVegetarianFood": True,
                         "location": {"latitude": 0, "longitude": 0}}, (0, 0))
        assert r.veg_options and not r.pure_veg


# --- filters ---------------------------------------------------------------------------------
def _mk(i, **kw):
    base = dict(id=str(i), name=str(i), lat=0, lng=0, rating=4.5, user_rating_count=500,
                distance_m=100, open_now=True)
    base.update(kw)
    return Restaurant(**base)


def test_diet_filter():
    rs = [_mk(1, pure_veg=True), _mk(2, attrs={"servesVegetarianFood": True}), _mk(3)]
    assert [r.id for r in filter_candidates(rs, 1000)] == ["1", "2", "3"]
    assert [r.id for r in filter_candidates(rs, 1000, diet="veg_options")] == ["1", "2"]
    assert [r.id for r in filter_candidates(rs, 1000, diet="pure_veg")] == ["1"]


def test_skip_cuisines():
    rs = [_mk(1, cuisine="chinese"), _mk(2, cuisine="indian"),
          _mk(3, cuisine="unknown", types=["bar", "restaurant"])]
    assert in_cuisine_group(rs[0], "Chinese and Asian")
    out = filter_candidates(rs, 1000, skip_cuisines=["Chinese and Asian", "Bars"])
    assert [r.id for r in out] == ["2"]


# --- mood-driven search ----------------------------------------------------------------------
def test_search_runs_one_text_search_per_query_and_records_found_by():
    client = PlacesClient("k", session=MagicMock())
    a = lambda pid, src: parse_place({"id": pid, "location": {"latitude": 0, "longitude": 0}},
                                     (0, 0), src)
    client.search_nearby = MagicMock(return_value=[a("p1", NEARBY)])
    calls = []

    def fake_text(q, *args, **kw):
        calls.append(q)
        return {"restaurants": [a("p1", q)], "desserts": [a("p1", q), a("p2", q)],
                "quiet cafe": [a("p3", q)]}[q]

    client.search_text = MagicMock(side_effect=fake_text)
    out = {r.id: r for r in client.search(0, 0, 1000, queries=["desserts", "desserts",
                                                                "quiet cafe"])}
    assert calls == [GENERIC_QUERY, "desserts", "quiet cafe"]  # deduped
    assert out["p1"].found_by == [NEARBY, GENERIC_QUERY, "desserts"]
    assert out["p1"].matched_queries == ["desserts"]
    assert out["p3"].features(1000)["intent_match"] == 1.0
    assert parse_place({"id": "x", "location": {}}, (0, 0), NEARBY).features(1)[
        "intent_match"] == 0.0


def test_merge_results_keeps_first_and_unions_sources():
    a1 = Restaurant(id="a", name="first", lat=0, lng=0, found_by=["nearby"])
    a2 = Restaurant(id="a", name="second", lat=0, lng=0, found_by=["desserts"])
    (m,) = merge_results([a1, a2])
    assert m.name == "first" and m.found_by == ["nearby", "desserts"]


def test_search_queries_cravings_first_and_capped(cfg):
    assert search_queries(cfg, make_mood(neutral=1.0)) == []
    tired = search_queries(cfg, make_mood(**TIRED))
    assert tired == ["comfort food", "biryani"]
    q = search_queries(cfg, make_mood(**STRESSED), ["Sweet", "Spicy"])
    assert q[:2] == ["desserts", "ice cream"]  # the craving's own searches come first
    assert len(q) == cfg["retrieval"]["max_queries"]


def test_demo_search_marks_matches():
    found = {r.name: r.matched_queries for r in demo_search(17.385, 78.4867, ["desserts"])}
    assert found["Sugar Rush Desserts"] == ["desserts"]
    assert found["Biryani Junction"] == []


def test_calm_preset_is_not_low_energy(cfg):
    from ranker import active_rules
    assert [r["name"] for r in active_rules(cfg, make_mood(**CALM))] == ["content_calm"]


# --- ranking ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def candidates():
    return filter_candidates(demo_restaurants(), max_distance_m=MAX_M)


def _top(cfg, mood_kw, cravings=(), k=5, diet="any"):
    mood = make_mood(**mood_kw)
    rec = Recommender(cfg)
    places = demo_search(17.385, 78.4867, rec.queries(mood, cravings))
    cands = filter_candidates(places, MAX_M, diet=diet)
    return rec.rank(mood, cands, MAX_M, hour=20, k=k, cravings=cravings)


def test_different_moods_give_different_lists(cfg):
    moods = {"tired": TIRED, "upbeat": UPBEAT, "stressed": STRESSED, "calm": CALM}
    tops = {m: [t.restaurant.name for t in _top(cfg, kw)] for m, kw in moods.items()}
    overlap = [len(set(tops[a]) & set(tops[b])) for a, b in itertools.combinations(tops, 2)]
    assert sum(overlap) / len(overlap) <= 2.5, tops
    assert tops["upbeat"][0] != tops["tired"][0] != tops["stressed"][0]


def test_craving_changes_the_top_pick(cfg):
    sweet = _top(cfg, dict(neutral=1.0), ["Sweet"], k=1)[0]
    spicy = _top(cfg, dict(neutral=1.0), ["Spicy"], k=1)[0]
    assert sweet.features["dessert"] == 1.0 and "sweet" in sweet.why.lower()
    assert spicy.features["spicy"] == 1.0 and "spicy" in spicy.why.lower()


def test_pure_veg_diet_only_returns_pure_veg(cfg):
    top = _top(cfg, TIRED, diet="pure_veg")
    assert top and all(t.restaurant.pure_veg for t in top)


def test_intent_match_breaks_ties(cfg):
    f = {"rating": 4.5, "weighted_rating": 4.5, "closeness": 0.5}
    rr = Recommender(cfg).rules
    m = make_mood(neutral=1.0)
    assert rr.score(m, {**f, "intent_match": 1.0}) > rr.score(m, f)


def test_explain_prefers_stated_craving(cfg):
    from ranker import active_cravings, active_rules
    f = {"dessert": 1.0, "comfort_food": 1.0}
    why = explain(cfg, active_rules(cfg, make_mood(**TIRED)), f, 1.0,
                  active_cravings(cfg, ["Sweet"]))
    assert why.startswith("Something sweet")


def _ranked(name, cuisine, score):
    return Ranked(Restaurant(id=name, name=name, lat=0, lng=0, cuisine=cuisine), score, "",
                  "rules", {})


def test_diversify_caps_cuisine():
    items = [_ranked("a", "mandi", 5), _ranked("b", "mandi", 4.9), _ranked("c", "mandi", 4.8),
             _ranked("d", "cafe", 3.0), _ranked("e", "unknown", 2.9),
             _ranked("f", "unknown", 2.8)]
    out = [r.restaurant.id for r in diversify(items, 4, max_per_cuisine=2, repeat_penalty=0.4)]
    assert out[:1] == ["a"] and "c" not in out and out.count("b") == 1
    assert set(out) == {"a", "b", "d", "e"}


def test_diversify_fills_when_everything_is_capped():
    items = [_ranked(str(i), "mandi", 5 - i) for i in range(4)]
    assert len(diversify(items, 4, max_per_cuisine=2)) == 4


def test_top5_has_at_most_two_per_cuisine(cfg):
    for kw in (TIRED, UPBEAT, STRESSED, CALM):
        cuisines = [t.restaurant.cuisine for t in _top(cfg, kw)]
        assert max(cuisines.count(c) for c in set(cuisines)) <= 2


def test_bandit_is_added_to_rules_not_replacing_them(cfg, candidates):
    rec = Recommender(cfg)
    rec.retrain(simulate_events(60, seed=4))
    mood = make_mood(**TIRED)
    r = candidates[0]
    feats = r.features(MAX_M)
    from ranker import active_rules, context_vector
    expected = rec.rules.score(mood, feats, active_rules(cfg, mood)) + \
        rec.bandit_weight * rec.bandit.ucb(context_vector(mood, feats, 20))
    got = {t.restaurant.id: t.score for t in rec.rank(mood, candidates, MAX_M, 20, k=50)}
    assert got[r.id] == pytest.approx(expected)
