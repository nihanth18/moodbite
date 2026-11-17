"""Google Places API (New) client, restaurant feature tagging and demo data."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

BASE_URL = "https://places.googleapis.com/v1"
NEARBY_FIELDS = ",".join(
    [
        "places.id",
        "places.displayName",
        "places.types",
        "places.primaryType",
        "places.rating",
        "places.userRatingCount",
        "places.location",
        "places.currentOpeningHours.openNow",
        "places.formattedAddress",
    ]
)
INCLUDED_TYPES = [
    "restaurant",
    "cafe",
    "bakery",
    "bar",
    "meal_takeaway",
    "ice_cream_shop",
    "dessert_shop",
]
# "Best rated" settings. A Bayesian average pulls ratings with few reviews toward a prior,
# so a 4.9 from 5 reviews doesn't outrank a 4.6 from 3,000.
MIN_RATING = 4.0
MIN_REVIEWS = 30
RATING_PRIOR = 3.9  # typical rating of a restaurant on Google Maps
RATING_PRIOR_WEIGHT = 100  # the prior counts as this many reviews
TAGS = ("comfort_food", "quick_service", "lively", "quiet", "dessert", "familiar")

# --- tagging rules: place types + name keywords -------------------------------------------
TYPE_TAGS: dict[str, set[str]] = {
    "comfort_food": {
        "pizza_restaurant", "hamburger_restaurant", "ramen_restaurant", "american_restaurant",
        "diner", "fast_food_restaurant", "indian_restaurant", "korean_restaurant",
        "chinese_restaurant", "breakfast_restaurant", "brunch_restaurant",
    },
    "quick_service": {
        "fast_food_restaurant", "meal_takeaway", "sandwich_shop", "cafe", "coffee_shop",
        "food_court", "bakery", "bagel_shop", "donut_shop",
    },
    "lively": {
        "bar", "pub", "night_club", "bar_and_grill", "wine_bar", "sports_bar",
        "brewery", "karaoke", "brewpub",
    },
    "quiet": {"cafe", "tea_house", "fine_dining_restaurant", "coffee_shop", "book_store"},
    "dessert": {
        "dessert_shop", "dessert_restaurant", "ice_cream_shop", "bakery", "confectionery",
        "chocolate_shop", "donut_shop", "candy_store", "juice_shop",
    },
}
NAME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "comfort_food": (
        "biryani", "pizza", "burger", "ramen", "noodle", "dosa", "soup", "pho", "thali",
        "curry", "fried chicken", "dumpling", "momo", "mac", "khichdi", "paratha", "diner",
    ),
    "quick_service": ("express", "quick", "grab", "takeaway", "kiosk", "to go", "café", "cafe"),
    "lively": ("pub", "brewpub", "brewery", "brewing", "social", "taproom", "cantina", "club", "live", "beer"),
    "quiet": ("tea", "garden", "bistro", "library", "zen", "fine dining"),
    "dessert": (
        "dessert", "cake", "ice cream", "gelato", "waffle", "sweet", "mithai", "chocolate",
        "bakery", "patisserie", "kulfi", "donut", "creamery", "falooda",
    ),
}
FAMILIAR_CHAINS = (
    "mcdonald", "domino", "kfc", "subway", "starbucks", "pizza hut", "burger king",
    "haldiram", "cafe coffee day", "barbeque nation", "taco bell", "dunkin", "baskin",
)
FAMILIAR_MIN_REVIEWS = 1000
GENERIC_TYPES = {
    "restaurant", "food", "point_of_interest", "establishment", "store", "meal_delivery",
    "meal_takeaway", "bar", "cafe", "bakery",
}


@dataclass
class Restaurant:
    id: str
    name: str
    lat: float
    lng: float
    types: list[str] = field(default_factory=list)
    primary_type: str | None = None
    rating: float | None = None  # 1..5
    user_rating_count: int = 0
    open_now: bool | None = None
    address: str = ""
    distance_m: float = 0.0
    cuisine: str = "unknown"
    tags: dict[str, int] = field(default_factory=dict)

    @property
    def weighted_rating(self) -> float | None:
        """Bayesian average: (n*R + m*C) / (n + m)."""
        if self.rating is None:
            return None
        n, m = max(self.user_rating_count, 0), RATING_PRIOR_WEIGHT
        return (n * self.rating + m * RATING_PRIOR) / (n + m)

    @property
    def maps_url(self) -> str:
        url = f"https://www.google.com/maps/dir/?api=1&destination={self.lat},{self.lng}"
        if self.id and not self.id.startswith("demo-"):
            url += f"&destination_place_id={self.id}"
        return url

    def features(self, max_distance_m: float) -> dict[str, Any]:
        """Feature dict used by the ranker and stored in the feedback log."""
        closeness = 1.0 - min(self.distance_m / max(max_distance_m, 1.0), 1.0)
        wr = self.weighted_rating
        return {
            "place_id": self.id,
            "cuisine": self.cuisine,
            "rating": self.rating,
            "user_rating_count": self.user_rating_count,
            "weighted_rating": None if wr is None else round(wr, 4),
            "distance_m": round(self.distance_m, 1),
            "closeness": round(closeness, 4),
            **{t: int(self.tags.get(t, 0)) for t in TAGS},
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlacesError(RuntimeError):
    pass


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def infer_cuisine(types: list[str], primary_type: str | None, name: str = "") -> str:
    candidates = ([primary_type] if primary_type else []) + list(types)
    for t in candidates:
        if t and t.endswith("_restaurant") and t not in GENERIC_TYPES:
            return t[: -len("_restaurant")]
    for t in candidates:
        if t in {"cafe", "coffee_shop", "bakery", "bar", "pub", "ice_cream_shop", "dessert_shop",
                 "sandwich_shop", "tea_house"}:
            return t
    return "unknown"


def _has_word(text: str, phrase: str) -> bool:
    """Whole-word match so e.g. 'tea' does not match 'steak' and 'live' not 'olive'."""
    # optional plural 's' so 'dessert' matches 'Desserts', 'sweet' matches 'Sweets'
    return re.search(rf"(?<!\w){re.escape(phrase)}s?(?!\w)", text) is not None


def infer_tags(
    name: str, types: list[str], user_rating_count: int = 0, primary_type: str | None = None
) -> dict[str, int]:
    """Derive 0/1 mood-relevant tags from place types and name keywords."""
    all_types = set(types) | ({primary_type} if primary_type else set())
    lname = name.lower()
    tags = {}
    for tag in ("comfort_food", "quick_service", "lively", "quiet", "dessert"):
        by_type = bool(all_types & TYPE_TAGS[tag])
        by_name = any(_has_word(lname, kw) for kw in NAME_KEYWORDS[tag])
        tags[tag] = int(by_type or by_name)
    # Lively and quiet are contradictory; lively wins (bars that are also cafes are rarely quiet).
    if tags["lively"]:
        tags["quiet"] = 0
    tags["familiar"] = int(
        user_rating_count >= FAMILIAR_MIN_REVIEWS or any(_has_word(lname, c) for c in FAMILIAR_CHAINS)
    )
    return tags


def parse_place(raw: dict[str, Any], origin: tuple[float, float]) -> Restaurant:
    loc = raw.get("location", {})
    lat, lng = float(loc.get("latitude", 0.0)), float(loc.get("longitude", 0.0))
    name = raw.get("displayName", {}).get("text", "Unnamed")
    types = list(raw.get("types", []))
    primary = raw.get("primaryType")
    count = int(raw.get("userRatingCount", 0) or 0)
    return Restaurant(
        id=raw.get("id", ""),
        name=name,
        lat=lat,
        lng=lng,
        types=types,
        primary_type=primary,
        rating=raw.get("rating"),
        user_rating_count=count,
        open_now=raw.get("currentOpeningHours", {}).get("openNow"),
        address=raw.get("formattedAddress", ""),
        distance_m=haversine_m(origin[0], origin[1], lat, lng),
        cuisine=infer_cuisine(types, primary, name),
        tags=infer_tags(name, types, count, primary),
    )


HYDERABAD = (17.3850, 78.4867)


def parse_browser_location(
    geo: Any,
) -> tuple[float | None, float | None, float | None, str | None]:
    """Parse the browser Geolocation API result -> (lat, lng, accuracy_m, error).

    None means the browser hasn't answered yet (permission prompt still open).
    """
    if not geo:
        return None, None, None, None
    if not isinstance(geo, dict):
        return None, None, None, "Unexpected response from the browser."
    if geo.get("error"):
        err = geo["error"]
        code = err.get("code") if isinstance(err, dict) else None
        msgs = {1: "Location permission was denied. Allow it in your browser (and in Windows "
                   "Settings, under Privacy and Location), or enter a place name instead.",
                2: "Your device couldn't determine its location. Enter a place name instead.",
                3: "Getting your location timed out. Reload the page or enter a place name."}
        return None, None, None, msgs.get(code, f"Couldn't get your location: {err}")
    coords = geo.get("coords") or {}
    try:
        lat, lng = float(coords["latitude"]), float(coords["longitude"])
    except (KeyError, TypeError, ValueError):
        return None, None, None, "The browser didn't return coordinates."
    acc = coords.get("accuracy")
    return lat, lng, (float(acc) if acc is not None else None), None


def filter_candidates(
    restaurants: list[Restaurant],
    max_distance_m: float,
    min_rating: float = MIN_RATING,
    min_reviews: int = MIN_REVIEWS,
    require_open: bool = True,
) -> list[Restaurant]:
    """Keep only top-rated places that are open now and within distance (deduplicated)."""
    out, seen = [], set()
    for r in restaurants:
        if r.id in seen:
            continue
        if require_open and r.open_now is not True:
            continue
        if r.rating is None or r.rating < min_rating or r.user_rating_count < min_reviews:
            continue
        if r.distance_m > max_distance_m:
            continue
        seen.add(r.id)
        out.append(r)
    return out


class PlacesClient:
    def __init__(self, api_key: str, session: requests.Session | None = None, timeout: float = 10):
        if not api_key:
            raise PlacesError("GOOGLE_PLACES_API_KEY is not set (use --demo to run without it)")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout

    def _post(self, path: str, body: dict[str, Any], field_mask: str) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": field_mask,
        }
        try:
            resp = self.session.post(f"{BASE_URL}/{path}", json=body, headers=headers,
                                     timeout=self.timeout)
        except requests.RequestException as e:
            raise PlacesError(f"Network error calling Places API: {e}") from e
        if resp.status_code != 200:
            try:
                msg = resp.json().get("error", {}).get("message", resp.text)
            except ValueError:
                msg = resp.text
            raise PlacesError(f"Places API error {resp.status_code}: {msg}")
        return resp.json()

    def geocode(self, query: str) -> tuple[float, float]:
        """Resolve free text (e.g. 'Indiranagar, Bengaluru') to lat/lng via Text Search."""
        data = self._post("places:searchText", {"textQuery": query, "pageSize": 1},
                          "places.location")
        places = data.get("places") or []
        if not places:
            raise PlacesError(f"Could not find a location for {query!r}")
        loc = places[0]["location"]
        return float(loc["latitude"]), float(loc["longitude"])

    def search_nearby(
        self, lat: float, lng: float, radius_m: float, max_results: int = 20
    ) -> list[Restaurant]:
        body = {
            "includedTypes": INCLUDED_TYPES,
            "maxResultCount": max(1, min(max_results, 20)),
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": float(min(max(radius_m, 1.0), 50_000.0)),
                }
            },
        }
        data = self._post("places:searchNearby", body, NEARBY_FIELDS)
        return [parse_place(p, (lat, lng)) for p in data.get("places", [])]

    def search_top_rated(
        self, lat: float, lng: float, radius_m: float, min_rating: float = MIN_RATING
    ) -> list[Restaurant]:
        """Text Search for open places with a minimum rating near (lat, lng).

        Nearby Search returns the 20 most *popular* places, which at night are often closed.
        This filters on the server for open, highly rated places, so it finds good places
        the nearby search misses.
        """
        body = {
            "textQuery": "restaurants",
            "openNow": True,
            "minRating": float(min_rating),
            "pageSize": 20,
            "locationBias": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": float(min(max(radius_m, 1.0), 50_000.0)),
                }
            },
        }
        data = self._post("places:searchText", body, NEARBY_FIELDS)
        return [parse_place(p, (lat, lng)) for p in data.get("places", [])]

    def search(self, lat: float, lng: float, radius_m: float,
               min_rating: float = MIN_RATING) -> list[Restaurant]:
        """Merge nearby and top-rated results (deduplicated by place id)."""
        merged: dict[str, Restaurant] = {}
        for r in self.search_nearby(lat, lng, radius_m) + self.search_top_rated(
                lat, lng, radius_m, min_rating):
            merged.setdefault(r.id, r)
        return list(merged.values())


# --- demo mode -----------------------------------------------------------------------------
_DEMO = [
    # name, types, rating, reviews, d_north_m, d_east_m, open_now
    ("Biryani Junction", ["indian_restaurant", "restaurant"], 4.4, 2300, 300, 150, True),
    ("The Hoppy Tap Brewpub", ["bar", "pub", "restaurant"], 4.3, 870, -500, 600, True),
    ("Irani Chai Tea House", ["tea_house", "cafe"], 4.6, 210, 200, -400, True),
    ("Sugar Rush Desserts", ["dessert_shop", "bakery"], 4.5, 640, -150, -100, True),
    ("Express Wraps", ["fast_food_restaurant", "meal_takeaway"], 4.1, 120, 80, 90, True),
    ("Mama's Ramen", ["ramen_restaurant", "restaurant"], 4.7, 950, 900, 300, True),
    ("Domino's Pizza", ["pizza_restaurant", "meal_takeaway"], 3.8, 4100, -300, -700, True),
    ("Saffron Fine Dining", ["fine_dining_restaurant", "indian_restaurant"], 4.8, 520, 1300, 0,
     True),
    ("Gelato Garden", ["ice_cream_shop"], 4.4, 330, 450, -950, True),
    ("Social Taco Cantina", ["mexican_restaurant", "bar"], 4.2, 1500, -1100, 400, True),
    ("Green Bowl Salads", ["vegan_restaurant", "restaurant"], 4.1, 180, 600, 700, True),
    ("Dosa Corner", ["south_indian_restaurant", "restaurant"], 4.5, 3000, -700, -200, True),
    ("Midnight Diner", ["diner", "american_restaurant"], 4.0, 410, 1800, -500, True),
    ("Closed Sushi Co.", ["sushi_restaurant"], 4.6, 700, 250, 250, False),
    ("Brand New Bistro", ["french_restaurant", "cafe"], 4.9, 6, -400, 1200, True),
    ("Haleem House", ["middle_eastern_restaurant", "restaurant"], 4.6, 1800, 700, -600, True),
]


def demo_restaurants(lat: float = HYDERABAD[0], lng: float = HYDERABAD[1]) -> list[Restaurant]:
    """Deterministic fake restaurants around (lat, lng). Used by --demo and tests."""
    out = []
    for i, (name, types, rating, reviews, dn, de, open_now) in enumerate(_DEMO):
        plat = lat + dn / 111_320.0
        plng = lng + de / (111_320.0 * math.cos(math.radians(lat)))
        raw = {
            "id": f"demo-{i}",
            "displayName": {"text": name},
            "types": types,
            "primaryType": types[0],
            "rating": rating,
            "userRatingCount": reviews,
            "location": {"latitude": plat, "longitude": plng},
            "currentOpeningHours": {"openNow": open_now},
            "formattedAddress": "Demo Street",
        }
        out.append(parse_place(raw, (lat, lng)))
    return out
