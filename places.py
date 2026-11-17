"""Google Places API (New) client, restaurant feature tagging, filters and demo data."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import requests

BASE_URL = "https://places.googleapis.com/v1"

# Boolean attributes from Places API (New). They make tags much more accurate than guessing
# from names, but they are billed at a higher tier than basic fields.
ATTRIBUTE_FIELDS = (
    "servesDessert", "servesCoffee", "servesCocktails", "servesBeer", "servesWine",
    "servesVegetarianFood", "servesBreakfast", "liveMusic", "outdoorSeating", "goodForGroups",
    "goodForWatchingSports", "takeout", "dineIn", "reservable",
)
PLACE_FIELDS = ",".join(
    "places." + f for f in (
        "id", "displayName", "types", "primaryType", "rating", "userRatingCount", "location",
        "currentOpeningHours.openNow", "formattedAddress", "editorialSummary",
        *ATTRIBUTE_FIELDS,
    )
)
NEARBY_FIELDS = PLACE_FIELDS  # kept for backwards compatibility
INCLUDED_TYPES = [
    "restaurant", "cafe", "bakery", "bar", "meal_takeaway", "ice_cream_shop", "dessert_shop",
]
GENERIC_QUERY = "restaurants"
NEARBY = "nearby"

# "Best rated" settings. A Bayesian average pulls ratings with few reviews toward a prior,
# so a 4.9 from 5 reviews doesn't outrank a 4.6 from 3,000.
MIN_RATING = 4.0
MIN_REVIEWS = 30
RATING_PRIOR = 3.9  # typical rating of a restaurant on Google Maps
RATING_PRIOR_WEIGHT = 100  # the prior counts as this many reviews

TAGS = ("comfort_food", "quick_service", "lively", "quiet", "dessert", "familiar",
        "outdoor", "spicy", "light", "coffee")
STRONG, WEAK, FAINT = 1.0, 0.5, 0.3

# --- tagging evidence ------------------------------------------------------------------------
TYPE_TAGS: dict[str, set[str]] = {
    "comfort_food": {
        "pizza_restaurant", "hamburger_restaurant", "ramen_restaurant", "american_restaurant",
        "diner", "fast_food_restaurant", "indian_restaurant", "korean_restaurant",
        "chinese_restaurant", "breakfast_restaurant", "brunch_restaurant",
        "south_indian_restaurant", "north_indian_restaurant", "family_restaurant",
    },
    "quick_service": {
        "fast_food_restaurant", "meal_takeaway", "sandwich_shop", "food_court", "bagel_shop",
        "donut_shop",
    },
    "lively": {
        "bar", "pub", "night_club", "bar_and_grill", "wine_bar", "sports_bar", "brewery",
        "karaoke", "brewpub", "cocktail_bar", "lounge_bar",
    },
    "quiet": {"tea_house", "fine_dining_restaurant", "coffee_shop", "book_store"},
    "dessert": {
        "dessert_shop", "dessert_restaurant", "ice_cream_shop", "bakery", "confectionery",
        "chocolate_shop", "donut_shop", "candy_store", "cake_shop",
    },
    "spicy": {
        "indian_restaurant", "south_indian_restaurant", "north_indian_restaurant",
        "thai_restaurant", "mexican_restaurant", "korean_restaurant", "sichuan_restaurant",
    },
    "light": {
        "vegan_restaurant", "vegetarian_restaurant", "salad_shop", "juice_shop",
        "japanese_restaurant", "sushi_restaurant", "mediterranean_restaurant",
        "health_food_restaurant", "acai_shop",
    },
    "coffee": {"cafe", "coffee_shop", "tea_house"},
}
# Weak type evidence (a cafe is often, but not always, quiet).
WEAK_TYPE_TAGS: dict[str, set[str]] = {
    "quiet": {"cafe"},
    "quick_service": {"cafe", "bakery"},
    "light": {"cafe", "breakfast_restaurant"},
}
KEYWORDS: dict[str, tuple[str, ...]] = {
    "comfort_food": (
        "biryani", "pizza", "burger", "ramen", "noodle", "dosa", "soup", "pho", "thali",
        "curry", "fried chicken", "dumpling", "momo", "khichdi", "paratha", "diner", "haleem",
        "comfort", "homestyle", "home-style", "homey", "hearty", "mandi", "meals",
    ),
    "quick_service": ("express", "quick", "grab", "takeaway", "kiosk", "to go", "fast food"),
    "lively": (
        "pub", "brewpub", "brewery", "brewing", "social", "taproom", "cantina", "club", "live",
        "beer", "lively", "buzzing", "vibrant", "party", "rooftop", "dj", "cocktail", "lounge",
    ),
    "quiet": (
        "tea", "garden", "bistro", "library", "zen", "fine dining", "quiet", "cozy", "cosy",
        "calm", "peaceful", "serene", "low-key", "relaxed", "tranquil",
    ),
    "dessert": (
        "dessert", "cake", "ice cream", "gelato", "waffle", "sweet", "mithai", "chocolate",
        "bakery", "patisserie", "kulfi", "donut", "creamery", "falooda", "pastry", "brownie",
    ),
    "outdoor": ("garden", "rooftop", "terrace", "outdoor", "patio", "al fresco", "lakeside",
                "poolside", "open-air"),
    "spicy": (
        "biryani", "andhra", "chettinad", "mandi", "haleem", "hyderabadi", "spicy", "schezwan",
        "sichuan", "chilli", "chili", "kebab", "tandoor", "military", "pepper", "rayalaseema",
    ),
    # Not "light" or "fresh": descriptions say "light-filled space", "freshly made".
    "light": ("salad", "healthy", "bowl", "juice", "organic", "vegan", "smoothie", "poke",
              "millet", "keto"),
    "coffee": ("coffee", "cafe", "café", "chai", "tea", "roast", "espresso", "brew"),
}
FAMILIAR_CHAINS = (
    "mcdonald", "domino", "kfc", "subway", "starbucks", "pizza hut", "burger king",
    "haldiram", "cafe coffee day", "barbeque nation", "taco bell", "dunkin", "baskin",
    "paradise", "bawarchi", "chutneys", "karachi bakery",
)
FAMILIAR_MIN_REVIEWS = 1000
PURE_VEG_TYPES = {"vegetarian_restaurant", "vegan_restaurant"}
PURE_VEG_WORDS = ("pure veg", "veg", "vegetarian", "vegan", "udupi", "sattvic", "jain")
NON_VEG_WORDS = ("non veg", "non-veg", "nonveg", "chicken", "mutton", "fish", "mandi",
                 "kebab", "biryani", "meat", "egg")
GENERIC_TYPES = {
    "restaurant", "food", "point_of_interest", "establishment", "store", "meal_delivery",
    "meal_takeaway", "bar", "cafe", "bakery", "food_store",
}
CUISINE_GROUPS: dict[str, set[str]] = {
    "Fast food": {"fast_food", "hamburger", "pizza", "sandwich_shop", "chicken"},
    "Chinese and Asian": {"chinese", "asian", "thai", "japanese", "korean", "sushi", "ramen",
                          "vietnamese", "indonesian", "sichuan"},
    "Continental": {"italian", "european", "french", "mediterranean", "american", "spanish",
                    "greek", "steak_house"},
    "Middle Eastern": {"middle_eastern", "lebanese", "turkish", "afghani", "persian"},
    "Bakery and desserts": {"bakery", "dessert_shop", "ice_cream_shop", "cake_shop", "dessert",
                            "confectionery", "donut_shop"},
    "Cafés": {"cafe", "coffee_shop", "tea_house"},
    "Bars": {"bar", "pub", "wine_bar", "bar_and_grill", "cocktail_bar"},
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
    tags: dict[str, float] = field(default_factory=dict)
    attrs: dict[str, bool] = field(default_factory=dict)
    summary: str = ""
    pure_veg: bool = False
    found_by: list[str] = field(default_factory=list)  # searches that returned this place

    @property
    def weighted_rating(self) -> float | None:
        """Bayesian average: (n*R + m*C) / (n + m)."""
        if self.rating is None:
            return None
        n, m = max(self.user_rating_count, 0), RATING_PRIOR_WEIGHT
        return (n * self.rating + m * RATING_PRIOR) / (n + m)

    @property
    def veg_options(self) -> bool:
        return self.pure_veg or bool(self.attrs.get("servesVegetarianFood"))

    @property
    def matched_queries(self) -> list[str]:
        """Mood/craving searches (not the generic ones) that found this place."""
        return [q for q in self.found_by if q not in (GENERIC_QUERY, NEARBY)]

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
            "intent_match": 1.0 if self.matched_queries else 0.0,
            "pure_veg": int(self.pure_veg),
            **{t: float(self.tags.get(t, 0.0)) for t in TAGS},
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
                 "sandwich_shop", "tea_house", "cake_shop"}:
            return t
    return "unknown"


def _has_word(text: str, phrase: str) -> bool:
    """Whole-word match so e.g. 'tea' does not match 'steak' and 'live' not 'olive'."""
    # optional plural 's' so 'dessert' matches 'Desserts', 'sweet' matches 'Sweets'
    return re.search(rf"(?<!\w){re.escape(phrase)}s?(?!\w)", text) is not None


def _any_word(text: str, words: Iterable[str]) -> bool:
    return any(_has_word(text, w) for w in words)


def infer_tags(
    name: str,
    types: list[str],
    user_rating_count: int = 0,
    primary_type: str | None = None,
    attrs: dict[str, bool] | None = None,
    summary: str = "",
) -> dict[str, float]:
    """Mood-relevant tags in [0, 1] from place types, Google attributes, name and summary.

    1.0 = strong evidence (type, name/description keyword, specific attribute such as live
    music); 0.5 / 0.3 = weak evidence (e.g. 'serves dessert', which most restaurants do).
    """
    attrs = attrs or {}
    all_types = set(types) | ({primary_type} if primary_type else set())
    text = f"{name} {summary}".lower()
    tags = {t: 0.0 for t in TAGS}

    def bump(tag: str, value: float) -> None:
        tags[tag] = max(tags[tag], value)

    for tag in TAGS:
        if all_types & TYPE_TAGS.get(tag, set()) or _any_word(text, KEYWORDS.get(tag, ())):
            bump(tag, STRONG)
        elif all_types & WEAK_TYPE_TAGS.get(tag, set()):
            bump(tag, WEAK)

    if attrs.get("liveMusic") or attrs.get("servesCocktails") or attrs.get(
            "goodForWatchingSports"):
        bump("lively", STRONG)
    elif attrs.get("servesBeer"):
        bump("lively", WEAK)
    if attrs.get("outdoorSeating"):
        bump("outdoor", STRONG)
    if attrs.get("servesDessert"):
        bump("dessert", FAINT)
    if attrs.get("servesCoffee"):
        bump("coffee", FAINT)
    if attrs.get("takeout") and not attrs.get("reservable"):
        bump("quick_service", WEAK)

    if tags["lively"] >= STRONG:  # a lively place isn't a quiet one
        tags["quiet"] = 0.0
    if user_rating_count >= FAMILIAR_MIN_REVIEWS or _any_word(name.lower(), FAMILIAR_CHAINS):
        tags["familiar"] = STRONG
    return tags


def infer_pure_veg(name: str, types: list[str], summary: str = "") -> bool:
    text = f"{name} {summary}".lower()
    if _any_word(text, NON_VEG_WORDS):
        return False
    return bool(set(types) & PURE_VEG_TYPES) or _any_word(name.lower(), PURE_VEG_WORDS) or \
        "pure vegetarian" in text or "all-vegetarian" in text


def parse_place(raw: dict[str, Any], origin: tuple[float, float],
                found_by: str | None = None) -> Restaurant:
    loc = raw.get("location", {})
    lat, lng = float(loc.get("latitude", 0.0)), float(loc.get("longitude", 0.0))
    name = raw.get("displayName", {}).get("text", "Unnamed")
    types = list(raw.get("types", []))
    primary = raw.get("primaryType")
    count = int(raw.get("userRatingCount", 0) or 0)
    attrs = {k: bool(raw[k]) for k in ATTRIBUTE_FIELDS if k in raw}
    summary = (raw.get("editorialSummary") or {}).get("text", "")
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
        tags=infer_tags(name, types, count, primary, attrs, summary),
        attrs=attrs,
        summary=summary,
        pure_veg=infer_pure_veg(name, types, summary),
        found_by=[found_by] if found_by else [],
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


def in_cuisine_group(r: Restaurant, group: str) -> bool:
    keys = CUISINE_GROUPS.get(group, set())
    names = {r.cuisine} | {t[: -len("_restaurant")] if t.endswith("_restaurant") else t
                           for t in r.types + ([r.primary_type] if r.primary_type else [])}
    return bool(names & keys)


def filter_candidates(
    restaurants: list[Restaurant],
    max_distance_m: float,
    min_rating: float = MIN_RATING,
    min_reviews: int = MIN_REVIEWS,
    require_open: bool = True,
    diet: str = "any",
    skip_cuisines: Iterable[str] = (),
) -> list[Restaurant]:
    """Keep open, top-rated places within distance that fit the diet (deduplicated).

    diet: 'any' | 'veg_options' (serves vegetarian food) | 'pure_veg' (vegetarian-only place)
    skip_cuisines: names from CUISINE_GROUPS to leave out.
    """
    skip = list(skip_cuisines)
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
        if diet == "pure_veg" and not r.pure_veg:
            continue
        if diet == "veg_options" and not r.veg_options:
            continue
        if any(in_cuisine_group(r, g) for g in skip):
            continue
        seen.add(r.id)
        out.append(r)
    return out


def _bounds(lat: float, lng: float, radius_m: float) -> dict[str, Any]:
    dlat = radius_m / 111_320.0
    dlng = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.01))
    return {"rectangle": {"low": {"latitude": lat - dlat, "longitude": lng - dlng},
                          "high": {"latitude": lat + dlat, "longitude": lng + dlng}}}


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
        data = self._post("places:searchNearby", body, PLACE_FIELDS)
        return [parse_place(p, (lat, lng), NEARBY) for p in data.get("places", [])]

    def search_text(
        self, query: str, lat: float, lng: float, radius_m: float,
        min_rating: float = MIN_RATING,
    ) -> list[Restaurant]:
        """Open places matching `query` with a minimum rating, inside the search area.

        Uses a rectangle restriction (Text Search doesn't accept a circle restriction); the
        exact distance is enforced afterwards by filter_candidates.
        """
        body = {
            "textQuery": query,
            "openNow": True,
            "minRating": float(min_rating),
            "pageSize": 20,
            "locationRestriction": _bounds(lat, lng, min(max(radius_m, 1.0), 50_000.0)),
        }
        data = self._post("places:searchText", body, PLACE_FIELDS)
        return [parse_place(p, (lat, lng), query) for p in data.get("places", [])]

    def search_top_rated(self, lat: float, lng: float, radius_m: float,
                         min_rating: float = MIN_RATING) -> list[Restaurant]:
        return self.search_text(GENERIC_QUERY, lat, lng, radius_m, min_rating)

    def search(self, lat: float, lng: float, radius_m: float, min_rating: float = MIN_RATING,
               queries: Iterable[str] = ()) -> list[Restaurant]:
        """Nearby + top-rated + one search per mood/craving query, merged by place id.

        Each place records every search that returned it (`found_by`), so the ranker knows
        which places were found specifically for this mood.
        """
        results = self.search_nearby(lat, lng, radius_m)
        results += self.search_top_rated(lat, lng, radius_m, min_rating)
        for q in dict.fromkeys(queries):  # dedupe, keep order
            if q and q != GENERIC_QUERY:
                results += self.search_text(q, lat, lng, radius_m, min_rating)
        return merge_results(results)


def merge_results(results: list[Restaurant]) -> list[Restaurant]:
    merged: dict[str, Restaurant] = {}
    for r in results:
        if r.id in merged:
            for q in r.found_by:
                if q not in merged[r.id].found_by:
                    merged[r.id].found_by.append(q)
        else:
            merged[r.id] = r
    return list(merged.values())


# --- demo mode -----------------------------------------------------------------------------
_DEMO = [
    # name, types, rating, reviews, d_north_m, d_east_m, open_now, attributes, summary
    ("Biryani Junction", ["indian_restaurant", "restaurant"], 4.4, 2300, 300, 150, True,
     {"servesVegetarianFood": True, "takeout": True}, "Hyderabadi dum biryani and haleem."),
    ("The Hoppy Tap Brewpub", ["bar", "pub", "restaurant"], 4.3, 870, -500, 600, True,
     {"liveMusic": True, "servesBeer": True, "goodForGroups": True}, ""),
    ("Irani Chai Tea House", ["tea_house", "cafe"], 4.6, 210, 200, -400, True,
     {"servesCoffee": True}, "A calm, cozy spot for Irani chai and Osmania biscuits."),
    ("Sugar Rush Desserts", ["dessert_shop", "bakery"], 4.5, 640, -150, -100, True, {}, ""),
    ("Express Wraps", ["fast_food_restaurant", "meal_takeaway"], 4.1, 120, 80, 90, True,
     {"takeout": True}, ""),
    ("Mama's Ramen", ["ramen_restaurant", "restaurant"], 4.7, 950, 900, 300, True, {}, ""),
    ("Domino's Pizza", ["pizza_restaurant", "meal_takeaway"], 3.8, 4100, -300, -700, True,
     {"takeout": True}, ""),
    ("Saffron Fine Dining", ["fine_dining_restaurant", "indian_restaurant"], 4.8, 520, 1300, 0,
     True, {"reservable": True, "servesVegetarianFood": True}, ""),
    ("Gelato Garden", ["ice_cream_shop"], 4.4, 330, 450, -950, True, {"outdoorSeating": True},
     ""),
    ("Social Taco Cantina", ["mexican_restaurant", "bar"], 4.2, 1500, -1100, 400, True,
     {"servesCocktails": True}, ""),
    ("Green Bowl Salads", ["vegan_restaurant", "restaurant"], 4.1, 180, 600, 700, True,
     {"servesVegetarianFood": True}, "Fresh salads, grain bowls and juices."),
    ("Dosa Corner", ["south_indian_restaurant", "restaurant"], 4.5, 3000, -700, -200, True,
     {"servesVegetarianFood": True, "servesCoffee": True}, ""),
    ("Midnight Diner", ["diner", "american_restaurant"], 4.0, 410, 1800, -500, True, {}, ""),
    ("Closed Sushi Co.", ["sushi_restaurant"], 4.6, 700, 250, 250, False, {}, ""),
    ("Brand New Bistro", ["french_restaurant", "cafe"], 4.9, 6, -400, 1200, True, {}, ""),
    ("Haleem House", ["middle_eastern_restaurant", "restaurant"], 4.6, 1800, 700, -600, True,
     {}, "Spicy haleem and mandi."),
    ("Udupi Pure Veg Tiffins", ["vegetarian_restaurant", "restaurant"], 4.5, 2600, -250, 450,
     True, {"servesVegetarianFood": True, "servesCoffee": True}, ""),
    ("Rooftop Garden Kitchen", ["restaurant"], 4.4, 980, 1100, 900, True,
     {"liveMusic": True, "outdoorSeating": True, "servesCocktails": True},
     "Rooftop dining with live music and city views."),
    ("Andhra Spice Meals", ["indian_restaurant", "restaurant"], 4.3, 1200, -900, -800, True,
     {"servesVegetarianFood": True}, "Fiery Andhra meals and Rayalaseema chicken."),
]


def demo_restaurants(lat: float = HYDERABAD[0], lng: float = HYDERABAD[1]) -> list[Restaurant]:
    """Deterministic fake restaurants around (lat, lng). Used by --demo and tests."""
    out = []
    for i, (name, types, rating, reviews, dn, de, open_now, attrs, summary) in enumerate(_DEMO):
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
            "editorialSummary": {"text": summary} if summary else {},
            **attrs,
        }
        out.append(parse_place(raw, (lat, lng), NEARBY))
    return out


_QUERY_STOPWORDS = {"food", "restaurant", "restaurants", "place", "places", "meals", "the"}


def demo_search(lat: float, lng: float, queries: Iterable[str] = ()) -> list[Restaurant]:
    """Demo stand-in for PlacesClient.search: marks which mood queries each place 'matches'."""
    places = demo_restaurants(lat, lng)
    for r in places:
        blob = " ".join([r.name, r.summary, *r.types, *[t for t, v in r.tags.items() if v >= 1]])
        blob = blob.lower().replace("_", " ")
        for q in queries:
            words = [w for w in q.lower().split() if w not in _QUERY_STOPWORDS]
            if words and any(_has_word(blob, w) for w in words):
                r.found_by.append(q)
    return places
