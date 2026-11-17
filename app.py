"""Mood-based restaurant recommendations (Streamlit UI).

    streamlit run app.py              # live mode (needs GOOGLE_PLACES_API_KEY in .env or secrets)
    streamlit run app.py -- --demo    # demo mode with fake restaurants, no API key needed
"""
from __future__ import annotations

import html
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit_js_eval import streamlit_js_eval

from emotion import EMOTIONS, EmotionAnalyzer, MoodVector, decode_image
from places import (
    HYDERABAD, PlacesClient, PlacesError, demo_restaurants, filter_candidates, haversine_m,
    parse_browser_location,
)
from ranker import Recommender, load_rules
from storage import Storage
from train import train_baseline

APP_DIR = Path(__file__).parent
load_dotenv(APP_DIR / ".env")


def _api_key() -> str:
    key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
    if not key:
        try:  # Streamlit Community Cloud / .streamlit/secrets.toml
            key = str(st.secrets.get("GOOGLE_PLACES_API_KEY", "")).strip()
        except Exception:
            key = ""
    # The placeholder in .env / .env.example counts as "no key", so the app stays in demo mode
    return "" if key.lower() in ("", "your-api-key-here") else key


API_KEY = _api_key()
DEMO_DEFAULT = "--demo" in sys.argv[1:] or os.getenv("MOODBITE_DEMO") == "1" or not API_KEY

# Ask for a GPS-grade fix (enableHighAccuracy) and watch for up to 10 s, keeping the most
# accurate reading; stop early once accuracy is <= 20 m. The first fix is often coarse.
HIGH_ACCURACY_GEO_JS = """
new Promise((resolve) => {
  if (!navigator.geolocation) {
    resolve({error: {code: 2, message: "Geolocation is not supported by this browser"}});
    return;
  }
  let best = null, done = false, id = null, timer = null;
  const finish = (fallback) => {
    if (done) return;
    done = true;
    if (id !== null) navigator.geolocation.clearWatch(id);
    clearTimeout(timer);
    resolve(best || fallback || {error: {code: 3, message: "timeout"}});
  };
  id = navigator.geolocation.watchPosition(
    (p) => {
      const c = p.coords;
      if (!best || c.accuracy < best.coords.accuracy) {
        best = {coords: {latitude: c.latitude, longitude: c.longitude, accuracy: c.accuracy},
                timestamp: p.timestamp};
      }
      if (c.accuracy <= 20) finish();
    },
    (e) => { if (!best) finish({error: {code: e.code, message: e.message}}); },
    {enableHighAccuracy: true, maximumAge: 0, timeout: 15000}
  );
  timer = setTimeout(() => finish(), 10000);
})
"""

# Mood presets for people who skip the camera, or who want to correct it.
MOODS = {
    "Happy": ({"happy": .8, "neutral": .1, "surprise": .1}, .7, .5),
    "Calm": ({"happy": .5, "neutral": .5}, .4, -.2),
    "Tired": ({"neutral": .6, "sad": .4}, -.1, -.6),
    "Stressed": ({"angry": .3, "fear": .3, "neutral": .3, "disgust": .1}, -.4, .4),
    "Low": ({"sad": .7, "neutral": .3}, -.6, -.3),
    "Just hungry": ({"neutral": 1.0}, 0.0, 0.0),
}
TAG_LABELS = {"comfort_food": "Comfort food", "quick_service": "Quick",
              "lively": "Lively", "quiet": "Quiet", "dessert": "Desserts",
              "familiar": "Popular"}
RATING_OPTIONS = {"4.0+": 4.0, "4.3+": 4.3, "4.5+": 4.5}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap');

/* Text elements only. A broad selector like [class*="st-"] would also hit Streamlit's icon
   font, and the icons would then render as their names ("keyboard_arrow_down"). */
html, body, .stApp, p, li, label, input, textarea, button, h1, h2, h3, h4,
[data-testid="stMarkdownContainer"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
}
.block-container {max-width: 640px; padding-top: 2.5rem; padding-bottom: 5rem;}
header[data-testid="stHeader"], footer, .stAppDeployButton {display: none !important;}

/* square, hairline aesthetic */
button, input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"],
div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stExpander"] details,
div[data-testid="stCameraInput"] * {
  border-radius: 0 !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] {border-color: #111 !important;}
div[data-testid="stExpander"] details {border: 1px solid #111 !important;}
.stButton > button, .stLinkButton > a {
  border: 1px solid #111 !important; background: #fff !important; color: #111 !important;
  font-weight: 500; letter-spacing: .02em; box-shadow: none !important;
}
.stButton > button:hover, .stLinkButton > a:hover {background: #111 !important; color: #fff !important;}
.stButton > button[kind="primary"] {background: #111 !important; color: #fff !important;
  text-transform: uppercase; letter-spacing: .14em; font-size: .8rem; padding: .85rem 0;}
.stButton > button[kind="primary"]:hover {background: #333 !important;}
.stButton > button:disabled {background: #f3f3f3 !important; color: #aaa !important;
  border-color: #ddd !important;}
div[data-testid="stProgress"] > div > div > div > div {background-color: #111 !important;}

.x-top {font-size: .72rem; letter-spacing: .28em; text-transform: uppercase; color: #111;
  border-bottom: 1px solid #111; padding-bottom: 14px; margin-bottom: 8px;}
.x-lede {font-family: 'Instrument Serif', Georgia, serif; font-size: 2.6rem; line-height: 1.05;
  color: #111; margin: 18px 0 6px;}
.x-sub {color: #666; font-size: .95rem; margin-bottom: 8px;}
.x-step {display: flex; align-items: baseline; gap: 14px; margin: 44px 0 12px;
  border-top: 1px solid #111; padding-top: 12px;}
.x-step .n {font-size: .72rem; letter-spacing: .2em; color: #111;}
.x-step .t {font-size: .72rem; letter-spacing: .28em; text-transform: uppercase; color: #111;}
.x-mood {font-family: 'Instrument Serif', Georgia, serif; font-size: 2.2rem; line-height: 1.1;
  color: #111;}
.x-meta {color: #666; font-size: .85rem; margin-top: 4px;}
.x-note {border-left: 2px solid #111; padding: 6px 0 6px 12px; color: #333; font-size: .9rem;
  margin: 8px 0;}
.x-rank {font-size: .72rem; letter-spacing: .2em; color: #888;}
.x-name {font-family: 'Instrument Serif', Georgia, serif; font-size: 1.7rem; line-height: 1.15;
  color: #111; margin: 2px 0 6px;}
.x-facts {font-size: .85rem; color: #444; letter-spacing: .01em;}
.x-why {font-size: .95rem; color: #111; margin: 10px 0;}
.x-tag {display: inline-block; border: 1px solid #111; padding: 1px 8px; font-size: .68rem;
  letter-spacing: .14em; text-transform: uppercase; margin: 0 6px 6px 0; color: #111;}
.x-small {color: #888; font-size: .78rem;}
</style>
"""

st.set_page_config(page_title="Food for your mood", page_icon="○", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)


# --- cached resources ------------------------------------------------------------------------
@st.cache_resource
def get_config():
    return load_rules()


@st.cache_resource
def get_analyzer() -> EmotionAnalyzer:
    det = get_config().get("detection", {})
    return EmotionAnalyzer(det.get("min_face_confidence", 0.6),
                           det.get("min_emotion_confidence", 0.3))


@st.cache_resource
def get_storage() -> Storage:
    db = os.getenv("MOODBITE_DB") or "moodbite.db"
    return Storage(db if os.path.isabs(db) else APP_DIR / db)


def get_recommender() -> Recommender:
    if "recommender" not in st.session_state:
        rec = Recommender(get_config())
        rec.retrain(get_storage().load_events())
        st.session_state.recommender = rec
    return st.session_state.recommender


# --- helpers ---------------------------------------------------------------------------------
def preset(label: str) -> MoodVector:
    probs, v, a = MOODS[label]
    return MoodVector({e: probs.get(e, 0.0) for e in EMOTIONS}, v, a)


def set_mood(mood: MoodVector | None, status: str = "", message: str = "",
             label: str | None = None) -> None:
    st.session_state.mood = mood.to_dict() if mood else None
    st.session_state.mood_status = status
    st.session_state.mood_message = message
    st.session_state.mood_label = label
    st.session_state.results = None


def on_pick_mood() -> None:
    label = st.session_state.get("mood_pick")
    if label:
        set_mood(preset(label), "manual", "", label)
    elif st.session_state.mood_status == "manual":
        set_mood(None)


def on_fix_mood(key: str) -> None:
    label = st.session_state.get(key)
    camera = st.session_state.get("camera_mood")
    if label:
        set_mood(preset(label), "manual", "", label)
    elif camera:  # un-selected the correction, so go back to what the camera saw
        set_mood(MoodVector.from_dict(camera[0]), camera[1], camera[2])


def describe(mood: MoodVector) -> str:
    v, a = mood.valence, mood.arousal
    if v > 0.2 and a > 0.2:
        return "Upbeat and energetic"
    if v > 0.2:
        return "Calm and content"
    if v < -0.2 and a > 0.2:
        return "A little tense"
    if v < -0.2:
        return "A little low"
    if a < -0.2:
        return "Low on energy"
    return "Fairly neutral"


def step(n: int, title: str) -> None:
    st.markdown(f'<div class="x-step"><span class="n">{n:02d}</span>'
                f'<span class="t">{title}</span></div>', unsafe_allow_html=True)


def note(text: str, target=st) -> None:
    target.markdown(f'<div class="x-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def fmt_count(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def travel(dist_m: float) -> str:
    d = f"{dist_m / 1000:.1f} km" if dist_m >= 1000 else f"{dist_m:.0f} m"
    if dist_m <= 1500:
        return f"{d}, {max(1, round(dist_m / 80))} min walk"
    return f"{d}, {max(1, round(dist_m / 350))} min drive"


# --- state -----------------------------------------------------------------------------------
store = get_storage()
rec = get_recommender()
for key, default in [("mood", None), ("mood_status", ""), ("mood_message", ""),
                     ("mood_label", None), ("results", None), ("feedback", {}),
                     ("last_frame", None), ("geo_attempt", 0)]:
    st.session_state.setdefault(key, default)
demo = st.session_state.get("demo", DEMO_DEFAULT)

# --- header ----------------------------------------------------------------------------------
st.markdown('<div class="x-top">Food for your mood</div>'
            '<div class="x-lede">The best-rated places near you, chosen for how you feel.</div>',
            unsafe_allow_html=True)
if demo:
    st.markdown('<div class="x-sub">Demo mode. Restaurants are made up.</div>',
                unsafe_allow_html=True)

# --- 1. mood ---------------------------------------------------------------------------------
step(1, "Mood")
tab_photo, tab_pick = st.tabs(["Photo", "Choose"])

with tab_photo:
    consent = st.checkbox("I agree to have my facial expression read from one photo.")
    with st.expander("How the photo is handled"):
        st.markdown(
            "- The photo is analysed in memory on the server to estimate your expression: seven "
            "emotion scores, plus how positive and how energetic you seem.\n"
            "- It is never saved. It is discarded right after analysis, and only those nine "
            "numbers are kept.\n"
            "- Feedback you give is logged with those numbers to improve suggestions. You can "
            "delete it at any time in Settings.\n"
            "- Only you should be in the frame. If more than one face is detected, nothing is "
            "analysed.")
    if consent:
        frame = st.camera_input("Photo", label_visibility="collapsed")
        if frame is not None:
            data = frame.getvalue()
            frame_id = getattr(frame, "file_id", None) or hash(data)
            if frame_id != st.session_state.last_frame:
                st.session_state.last_frame = frame_id
                rgb = None
                try:
                    with st.spinner("Reading your expression"):
                        rgb = decode_image(data)
                        result = get_analyzer().analyze(rgb)
                except Exception as e:  # model download/load failures etc.
                    set_mood(None, "error", f"Couldn't run the emotion model: {e}")
                    st.session_state.camera_mood = None
                else:
                    set_mood(result.mood, result.status, result.message)
                    st.session_state.camera_mood = (
                        (result.mood.to_dict(), result.status, result.message)
                        if result.mood else None)
                finally:
                    rgb = data = None  # drop image references immediately

            # Expressions are an imperfect signal, so offer a one-tap correction. Corrections
            # are what gets logged, so the learner trains on the user's real mood.
            if st.session_state.get("camera_mood"):
                fix_key = f"fix-{st.session_state.last_frame}"
                st.pills("Not quite right? Choose what fits.", list(MOODS), key=fix_key,
                         on_change=on_fix_mood, args=(fix_key,))

with tab_pick:
    st.pills("How do you feel?", list(MOODS), key="mood_pick", on_change=on_pick_mood)

status, msg = st.session_state.mood_status, st.session_state.mood_message
if status in ("no_face", "multiple_faces", "low_confidence", "error"):
    note(msg)
mood = MoodVector.from_dict(st.session_state.mood) if st.session_state.mood else None
if mood:
    label = st.session_state.mood_label
    if label:
        head, sub = label, "Chosen by you"
    else:
        head = describe(mood)
        sub = f"Mostly {mood.top_emotion}, {mood.confidence:.0%} confidence"
    st.markdown(f'<div class="x-mood">{html.escape(head)}</div>'
                f'<div class="x-meta">{html.escape(sub)}</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    c1.progress((mood.valence + 1) / 2, text="Positivity")
    c2.progress((mood.arousal + 1) / 2, text="Energy")
    if status == "uncertain":
        note(msg)
    if not label:
        with st.expander("Emotion scores"):
            st.bar_chart(mood.probs, horizontal=True, height=220, color="#111111")

# --- 2. where ---------------------------------------------------------------------------------
step(2, "Location")
mode = st.segmented_control("Location", ["Near me", "Search an area"], default="Near me",
                            label_visibility="collapsed") or "Near me"
place_query, lat, lng = None, None, None
location_ready = True
if mode == "Near me":
    geo = streamlit_js_eval(js_expressions=HIGH_ACCURACY_GEO_JS,
                            key=f"geo-{st.session_state.geo_attempt}")
    lat, lng, acc, geo_err = parse_browser_location(geo)
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    if geo_err:
        note(geo_err, c1)
        location_ready = False
    elif lat is None:
        note("Finding your location. Allow access when your browser asks.", c1)
        location_ready = False
    else:
        acc_txt = f", accurate to {acc:.0f} m" if acc else ""
        c1.markdown(f'<div class="x-facts">Location found{acc_txt}</div>',
                    unsafe_allow_html=True)
    if geo is not None and c2.button("Refresh", use_container_width=True):
        st.session_state.geo_attempt += 1
        st.rerun()
    if lat is not None and acc and acc > 100:
        note("This is a rough fix. Turn on GPS or precise location, move near a window, "
             "then refresh.")
    if lat is not None and haversine_m(lat, lng, *HYDERABAD) > 40_000:
        note("This is far from Hyderabad. Laptops sometimes estimate location from the "
             "network. If it's wrong, search an area instead.")
else:
    place_query = st.text_input("Area, street or landmark", "Banjara Hills, Hyderabad")
    location_ready = bool(place_query.strip())

c1, c2 = st.columns(2)
max_km = c1.slider("Distance", 0.5, 10.0, 3.0, 0.5, format="%.1f km")
rating_label = c2.segmented_control("Minimum rating", list(RATING_OPTIONS),
                                    default="4.0+") or "4.0+"
min_rating = RATING_OPTIONS[rating_label]

st.write("")
find = st.button("Find places", type="primary", use_container_width=True,
                 disabled=mood is None or not location_ready)
if mood is None:
    st.markdown('<div class="x-small">Take a photo or choose a mood first.</div>',
                unsafe_allow_html=True)

if find:
    max_m = max_km * 1000
    try:
        with st.spinner("Searching"):
            if demo:
                if lat is None:
                    lat, lng = HYDERABAD
                places = demo_restaurants(lat, lng)
            else:
                client = PlacesClient(API_KEY)
                if lat is None:
                    lat, lng = client.geocode(place_query)
                places = client.search(lat, lng, max_m, min_rating)
        candidates = filter_candidates(places, max_m, min_rating=min_rating)
        weight = 0.5 if status == "uncertain" else 1.0
        st.session_state.results = rec.rank(mood, candidates, max_m, datetime.now().hour,
                                            k=5, mood_weight=weight)
        st.session_state.ranked_mood = mood.to_dict()
        st.session_state.search_desc = (f"Open now. Rated {rating_label}. "
                                        f"Within {max_km:g} km.")
        st.session_state.feedback = {}
        if not candidates:
            note(f"No open places rated {min_rating}+ within {max_km:g} km right now. "
                 "Try a longer distance or a lower minimum rating.")
    except PlacesError as e:
        note(str(e))

# --- 3. results ------------------------------------------------------------------------------
results = st.session_state.results
if results:
    step(3, "Suggestions")
    st.markdown(f'<div class="x-small">{html.escape(st.session_state.get("search_desc", ""))}'
                '</div>', unsafe_allow_html=True)
    for i, item in enumerate(results):
        r = item.restaurant
        with st.container(border=True):
            cuisine = r.cuisine.replace("_", " ").title() if r.cuisine != "unknown" else ""
            facts = [f"{r.rating:.1f} / 5 from {fmt_count(r.user_rating_count)} reviews"]
            facts += [html.escape(cuisine)] if cuisine else []
            facts.append(travel(r.distance_m))
            tags = "".join(f'<span class="x-tag">{TAG_LABELS[t]}</span>'
                           for t in TAG_LABELS if item.features.get(t))
            st.markdown(
                f'<div class="x-rank">{i + 1:02d}</div>'
                f'<div class="x-name">{html.escape(r.name)}</div>'
                f'<div class="x-facts">{" &nbsp;·&nbsp; ".join(facts)}</div>'
                f'<div class="x-why">{html.escape(item.why)}</div><div>{tags}</div>',
                unsafe_allow_html=True)
            given = st.session_state.feedback.get(r.id)
            b1, b2, b3 = st.columns(3)
            b1.link_button("Directions", r.maps_url, use_container_width=True)
            if given:
                b2.markdown('<div class="x-small" style="padding-top:.6rem">'
                            f'{"Enjoy your meal." if given == "going" else "Noted."}</div>',
                            unsafe_allow_html=True)
            else:
                for col, action, text in [(b2, "going", "Going"),
                                          (b3, "not_feeling_it", "Not for me")]:
                    if col.button(text, key=f"{action}-{r.id}", use_container_width=True):
                        store.log_feedback(st.session_state.ranked_mood, item.features, action,
                                           item.policy)
                        st.session_state.feedback[r.id] = action
                        st.rerun()

# --- settings --------------------------------------------------------------------------------
step(4 if results else 3, "Settings")
with st.expander("Settings and your data"):
    st.toggle("Demo mode", value=DEMO_DEFAULT, key="demo")
    if not st.session_state.get("demo", DEMO_DEFAULT) and not API_KEY:
        note("No Google Places API key is set (in .env or Streamlit secrets).")
    n_events = store.count()
    c1, c2 = st.columns(2)
    c1.metric("Feedback given", n_events)
    c2.metric("Ranking", "Learned" if rec.policy == "linucb" else "Rules",
              help=f"Learns from your feedback (LinUCB) after {rec.min_events} responses.")
    if st.button("Retrain from my feedback", use_container_width=True):
        events = store.load_events()
        rec.retrain(events)
        st.session_state.train_metrics = train_baseline(events)
        st.session_state.results = None
        st.rerun()
    if "train_metrics" in st.session_state:
        st.caption("Offline baseline (SGD logistic regression)")
        st.json(st.session_state.train_metrics, expanded=False)
    c1, c2 = st.columns(2)
    if c1.button("Forget my mood", use_container_width=True):
        set_mood(None)
        st.rerun()
    if c2.button("Delete my feedback", use_container_width=True):
        store.clear()
        rec.retrain([])
        st.session_state.pop("train_metrics", None)
        note("All logged feedback deleted.")
