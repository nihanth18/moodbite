from unittest.mock import MagicMock

import pytest
import requests

from places import (
    HYDERABAD, NEARBY_FIELDS, PlacesClient, parse_browser_location, PlacesError, Restaurant, demo_restaurants, filter_candidates,
    haversine_m, infer_cuisine, infer_tags, parse_place,
)

ORIGIN = (12.9716, 77.5946)


def raw_place(**kw):
    base = {
        "id": "p1",
        "displayName": {"text": "Test Place"},
        "types": ["restaurant"],
        "primaryType": "restaurant",
        "rating": 4.2,
        "userRatingCount": 100,
        "location": {"latitude": ORIGIN[0] + 0.001, "longitude": ORIGIN[1]},
        "currentOpeningHours": {"openNow": True},
        "formattedAddress": "1 Test St",
    }
    base.update(kw)
    return base


# --- tagging ---------------------------------------------------------------------------------
class TestTagging:
    def test_tags_from_types(self):
        t = infer_tags("Somewhere", ["ice_cream_shop"])
        assert t["dessert"] == 1 and t["lively"] == 0

    def test_tags_from_name_keywords(self):
        t = infer_tags("Hyderabadi Biryani House", ["restaurant"])
        assert t["comfort_food"] == 1

    def test_plural_keyword(self):
        assert infer_tags("Sweets & Treats", ["restaurant"])["dessert"] == 1

    def test_whole_word_matching_avoids_false_positives(self):
        t = infer_tags("Olive Steakhouse", ["restaurant"])
        assert t["lively"] == 0  # 'live' must not match 'olive'
        assert t["quiet"] == 0  # 'tea' must not match 'steak'

    def test_lively_overrides_quiet(self):
        t = infer_tags("Garden Pub", ["bar", "cafe"])
        assert t["lively"] == 1 and t["quiet"] == 0

    def test_familiar_by_chain_or_reviews(self):
        assert infer_tags("Domino's Pizza", ["pizza_restaurant"], 50)["familiar"] == 1
        assert infer_tags("Local Spot", ["restaurant"], 5000)["familiar"] == 1
        assert infer_tags("Local Spot", ["restaurant"], 10)["familiar"] == 0

    def test_quick_service(self):
        assert infer_tags("Wrap It", ["fast_food_restaurant"])["quick_service"] == 1

    def test_cuisine(self):
        assert infer_cuisine(["restaurant", "thai_restaurant"], "thai_restaurant") == "thai"
        assert infer_cuisine(["cafe", "restaurant"], "cafe") == "cafe"
        assert infer_cuisine(["restaurant"], None) == "unknown"


# --- parsing / filtering ---------------------------------------------------------------------
def test_parse_place_maps_fields():
    r = parse_place(raw_place(), ORIGIN)
    assert r.rating == 4.2 and r.user_rating_count == 100
    assert r.open_now is True
    assert 100 < r.distance_m < 125  # 0.001 deg latitude ≈ 111 m
    f = r.features(max_distance_m=1000)
    assert 0.85 < f["closeness"] < 0.9
    assert "price_level" not in f
    assert set(f) >= {"cuisine", "rating", "weighted_rating", "distance_m", "comfort_food",
                      "quick_service", "lively", "quiet", "dessert", "familiar"}


def test_parse_place_handles_missing_fields():
    r = parse_place({"id": "x", "location": {"latitude": 1, "longitude": 1}}, (1, 1))
    assert r.rating is None and r.open_now is None
    assert r.name == "Unnamed"


def test_haversine():
    assert haversine_m(0, 0, 0, 0) == 0
    assert 110_000 < haversine_m(0, 0, 1, 0) < 112_000


def test_filter_candidates_keeps_only_top_rated_open_nearby():
    def mk(i, rating, reviews=500, dist=500, open_now=True):
        return Restaurant(id=str(i), name=str(i), lat=0, lng=0, rating=rating,
                          user_rating_count=reviews, distance_m=dist, open_now=open_now)

    rs = [
        mk(1, 4.5),                    # keep
        mk(2, 3.9),                    # below 4.0
        mk(3, None),                   # unrated
        mk(4, 4.8, reviews=5),         # too few reviews to trust
        mk(5, 4.6, dist=5000),         # too far
        mk(6, 4.6, open_now=False),    # closed
        mk(7, 4.6, open_now=None),     # unknown hours
        mk(1, 4.5),                    # duplicate id
        mk(8, 4.0),                    # boundary: keep
    ]
    out = [r.id for r in filter_candidates(rs, max_distance_m=1000)]
    assert out == ["1", "8"]
    assert [r.id for r in filter_candidates(rs, 1000, min_rating=4.3)] == ["1"]


def test_weighted_rating_prefers_well_reviewed():
    few = Restaurant(id="a", name="a", lat=0, lng=0, rating=4.9, user_rating_count=5)
    many = Restaurant(id="b", name="b", lat=0, lng=0, rating=4.6, user_rating_count=3000)
    assert many.weighted_rating > few.weighted_rating
    assert Restaurant(id="c", name="c", lat=0, lng=0).weighted_rating is None


def test_maps_url():
    real = Restaurant(id="ChIJabc", name="x", lat=17.4, lng=78.4)
    assert "destination=17.4,78.4" in real.maps_url
    assert "destination_place_id=ChIJabc" in real.maps_url
    assert "place_id" not in demo_restaurants()[0].maps_url

def test_demo_restaurants_are_deterministic_and_tagged():
    a, b = demo_restaurants(), demo_restaurants()
    assert [r.name for r in a] == [r.name for r in b]
    assert len(a) >= 10
    by_name = {r.name: r for r in a}
    assert by_name["Sugar Rush Desserts"].tags["dessert"] == 1
    assert by_name["The Hoppy Tap Brewpub"].tags["lively"] == 1
    assert by_name["Irani Chai Tea House"].tags["quiet"] == 1
    assert by_name["Closed Sushi Co."].open_now is False


# --- mocked API client -----------------------------------------------------------------------
def mock_session(status=200, payload=None):
    resp = MagicMock(status_code=status)
    resp.json.return_value = payload if payload is not None else {}
    resp.text = "err"
    session = MagicMock()
    session.post.return_value = resp
    return session


def test_search_nearby_request_and_parse():
    session = mock_session(payload={"places": [raw_place(), raw_place(id="p2")]})
    client = PlacesClient("fake-key", session=session)
    out = client.search_nearby(*ORIGIN, radius_m=1500, max_results=50)

    assert [r.id for r in out] == ["p1", "p2"]
    url = session.post.call_args.args[0]
    kwargs = session.post.call_args.kwargs
    assert url.endswith("/places:searchNearby")
    assert kwargs["headers"]["X-Goog-Api-Key"] == "fake-key"
    assert kwargs["headers"]["X-Goog-FieldMask"] == NEARBY_FIELDS
    body = kwargs["json"]
    assert body["maxResultCount"] == 20  # clamped to API max
    assert body["locationRestriction"]["circle"]["radius"] == 1500.0
    assert "restaurant" in body["includedTypes"]


def test_search_nearby_empty_response():
    client = PlacesClient("k", session=mock_session(payload={}))
    assert client.search_nearby(*ORIGIN, 1000) == []


def test_api_error_raises_places_error():
    session = mock_session(status=403, payload={"error": {"message": "API key not valid"}})
    with pytest.raises(PlacesError, match="403.*API key not valid"):
        PlacesClient("bad", session=session).search_nearby(*ORIGIN, 1000)


def test_network_error_raises_places_error():
    session = MagicMock()
    session.post.side_effect = requests.ConnectionError("boom")
    with pytest.raises(PlacesError, match="Network error"):
        PlacesClient("k", session=session).search_nearby(*ORIGIN, 1000)


def test_search_top_rated_request():
    session = mock_session(payload={"places": [raw_place(id="t1")]})
    out = PlacesClient("k", session=session).search_top_rated(*ORIGIN, 2000, min_rating=4.3)
    assert [r.id for r in out] == ["t1"]
    assert session.post.call_args.args[0].endswith("/places:searchText")
    body = session.post.call_args.kwargs["json"]
    assert body["openNow"] is True and body["minRating"] == 4.3
    assert body["locationBias"]["circle"]["radius"] == 2000.0


def test_search_merges_and_dedupes():
    client = PlacesClient("k", session=mock_session())
    client.search_nearby = MagicMock(return_value=[parse_place(raw_place(id="a"), ORIGIN),
                                                   parse_place(raw_place(id="b"), ORIGIN)])
    client.search_top_rated = MagicMock(return_value=[parse_place(raw_place(id="b"), ORIGIN),
                                                      parse_place(raw_place(id="c"), ORIGIN)])
    assert [r.id for r in client.search(*ORIGIN, 1000)] == ["a", "b", "c"]


def test_geocode():
    session = mock_session(payload={"places": [{"location": {"latitude": 1.5,
                                                              "longitude": 2.5}}]})
    assert PlacesClient("k", session=session).geocode("somewhere") == (1.5, 2.5)
    assert session.post.call_args.args[0].endswith("/places:searchText")


def test_geocode_not_found():
    with pytest.raises(PlacesError, match="Could not find"):
        PlacesClient("k", session=mock_session(payload={})).geocode("nowhere")


class TestBrowserLocation:
    def test_pending(self):
        assert parse_browser_location(None) == (None, None, None, None)

    def test_success(self):
        geo = {"coords": {"latitude": 17.4126, "longitude": 78.4482, "accuracy": 35},
               "timestamp": 1}
        assert parse_browser_location(geo) == (17.4126, 78.4482, 35.0, None)

    def test_permission_denied(self):
        lat, _, _, err = parse_browser_location({"error": {"code": 1, "message": "denied"}})
        assert lat is None and "permission" in err.lower()

    def test_malformed(self):
        assert parse_browser_location({"coords": {}})[3] is not None
        assert parse_browser_location("nope")[3] is not None


def test_demo_defaults_to_hyderabad():
    r = demo_restaurants()
    assert all(haversine_m(r_.lat, r_.lng, *HYDERABAD) < 3000 for r_ in r)


def test_missing_api_key():
    with pytest.raises(PlacesError, match="--demo"):
        PlacesClient("")
