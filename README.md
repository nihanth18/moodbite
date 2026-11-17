# MoodBite 🍜

The best-rated restaurants open near you, picked for how you feel right now.
Take a selfie (or pick a mood), and MoodBite reads your expression, finds open places rated
4.0★+ nearby, ranks them for your mood, and learns from your feedback.

## Setup

Requires Python 3.11.

```bash
cd moodbite
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt   # app + tests (requirements.txt is the app alone)
cp .env.example .env                  # then set GOOGLE_PLACES_API_KEY
```

The API key needs **Places API (New)** enabled in Google Cloud. The app reads it from `.env`,
an environment variable, or Streamlit secrets. It is never hardcoded. If the key is missing, or
still the `your-api-key-here` placeholder, the app runs in demo mode.

## Run locally

```bash
streamlit run app.py               # live mode (Google Places)
streamlit run app.py -- --demo     # demo mode: made-up restaurants, no API key needed
python -m pytest -q                # tests (Places API is mocked)
python train.py                    # offline baseline on logged feedback
python train.py --simulate 300     # add 300 synthetic events first (try the bandit quickly)
```

### On your phone over home Wi-Fi

Phone browsers only allow the camera and GPS on HTTPS pages:

```bash
python scripts/make_cert.py        # one time; prints https://<your-PC-IP>:8501
streamlit run app.py --server.address 0.0.0.0 --server.sslCertFile certs/cert.pem \
    --server.sslKeyFile certs/key.pem
```

Open the printed URL on a phone on the same Wi-Fi. The certificate is self-signed, so the phone
warns once. Tap *Advanced → Proceed* (Chrome) or *Show Details → visit this website* (Safari).
If the page doesn't load, allow Python through Windows Firewall.

## Hosting (Streamlit Community Cloud, free)

Hosting gives you a real HTTPS link like `https://moodbite.streamlit.app` that works on any
phone, with no certificate warning and no PC left running.

1. **Put the code on GitHub.** A private repo is fine. `.gitignore` already keeps `.env`,
   `certs/`, `*.db` and `.venv/` out.
   ```bash
   cd moodbite
   git init && git add . && git commit -m "MoodBite"
   gh repo create moodbite --private --source . --push
   ```
   Without the `gh` CLI, create an empty repo on github.com, then
   `git remote add origin <url>` and `git push -u origin main`.
2. Go to **https://share.streamlit.io**, sign in with GitHub, and choose **Create app → Deploy a public app from GitHub**.
   Pick the repo, branch `main`, main file `app.py`.
3. Open **Advanced settings**:
   - **Python version:** 3.11
   - **Secrets:** paste `GOOGLE_PLACES_API_KEY = "your real key"`, plus `DATABASE_URL = "..."`
     if you set up a database (see below and `.streamlit/secrets.toml.example`)
4. Click **Deploy**. The first build takes a few minutes, because MediaPipe and ONNX are large.
   `packages.txt` installs the system libraries OpenCV needs.
5. **Lock it down.** In the app's **Settings → Sharing**, restrict viewers to specific email
   addresses. Anyone with a public link can otherwise run searches on your Google bill.
6. **Protect the key.** In Google Cloud → Credentials, restrict the key to *Places API (New)*
   and set a daily quota or budget alert. Each search makes 2–5 Places calls (nearby,
   top-rated, and up to 3 mood/craving searches, set by `retrieval.max_queries`). They request
   restaurant attributes (live music, outdoor seating, vegetarian, …), which Google bills at a
   higher tier than basic fields. Identical searches within 10 minutes are served from cache.

### Keep feedback permanently (hosted database)

Community Cloud's disk is temporary, so without a database the feedback log is wiped whenever
the app restarts, and the learned ranking never builds up. Any Postgres database works:

1. Create a free Postgres database, e.g. at **neon.tech** (or Supabase).
2. Copy its connection string. It looks like
   `postgresql://user:password@host/dbname?sslmode=require`.
3. Add it to the app's **Settings → Secrets** as `DATABASE_URL = "..."` and save. The app
   creates its `feedback` table on first use. Settings in the app then says
   "Feedback storage: Hosted database".

Locally, leave `DATABASE_URL` unset to use the SQLite file.

Things to know about Community Cloud:
- **Feedback resets on restart** unless `DATABASE_URL` is set (see above).
- **The app sleeps** after a period without visitors. The first visit after that takes about a minute to wake it.
- **Photos are processed on Streamlit's servers** (in memory, never saved), not on your PC.
  The in-app privacy note says "on the MoodBite server" for this reason.

Other options: any Docker host (Render, Railway, Fly.io, Google Cloud Run) works with
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0` and the same packages,
plus `GOOGLE_PLACES_API_KEY` as an environment variable.

## Using the app

1. **Mood.** Take a photo (after ticking consent) or choose a mood. After a photo you can
   correct the detected mood in one tap.
2. **Taste.** Diet (*Any*, *Veg options*, *Pure veg*), optional cravings (*Spicy*, *Sweet*,
   *Light*, *Hearty*, *Coffee*, *Drinks*), and cuisines to leave out.
3. **Location.** *Near me* uses your phone's GPS (high-accuracy fix, error margin shown,
   *Refresh* button), or search an area by name. Set how far you'll go (up to 20 km) and a
   minimum rating (4.0, 4.3 or 4.5).
4. **Suggestions.** Five open, top-rated places (at most two of any cuisine), each with rating,
   review count, distance, walk or drive time, Google's description when there is one, a
   one-line reason, tags, which mood search found it, a **Directions** button and
   **Going** / **Not for me** feedback.

Settings (demo mode, Going rate, retraining, deleting your data) are at the bottom of the page.

## How it works

```
photo ─► MediaPipe face detection ─► crop ─► HSEmotion (enet_b0_8_va_mtl, + mirrored copy)
          │ 0 faces / >1 face / low score → friendly message, nothing analysed
          ▼
     mood vector: 7 emotion probs + valence + arousal   (image discarded here)
          ▼
mood + cravings ─► up to 3 targeted searches (e.g. stressed → "desserts", "quiet cafe")
          ▼
Places: searchNearby + searchText("restaurants") + searchText(each query), all open now,
        rating ≥ 4.0, inside your distance ─► merge (remember which search found each place)
          ▼
filter: ≥ 30 reviews, diet, excluded cuisines ─► tags from Google attributes, types,
        name and description
          ▼
score = rules (rating, distance, mood, cravings, matched search) + LinUCB (after 20 events)
          ▼
top 5 with at most 2 per cuisine + one-line "why"
          ▼
"Going" / "Not for me" ─► SQLite or Postgres (numbers + restaurant features only)
```

| File | What it does |
|---|---|
| `app.py` | Streamlit UI (mobile-first; theme in `.streamlit/config.toml`). |
| `emotion.py` | MediaPipe face detection (Tasks API) and HSEmotion emotion, valence and arousal. Handles no face, multiple faces (refused: only the consenting user is analysed), low detection confidence, tiny faces, and uncertain expressions (used at half weight). |
| `places.py` | Places API (New) client (nearby, top-rated and per-query text searches), tagging, diet and cuisine filters, demo data. |
| `ranker.py` | Mood/craving searches, rule scoring from `config/mood_rules.yaml`, LinUCB, cuisine diversity, and the `Recommender` that combines them. |
| `storage.py` | Feedback log on SQLite or Postgres (`DATABASE_URL`), Going-rate metrics, "delete all my data". |
| `train.py` | Offline baseline: SGD logistic regression, prints accuracy and AUC against a majority-class baseline. |

**Mood decides what is searched.** Each mood rule and craving in `config/mood_rules.yaml` lists
Google searches (e.g. *stressed* → "desserts", "quiet cafe"; *Spicy* → "biryani", "andhra meals").
Up to `retrieval.max_queries` (3) run per search, cravings first, so different moods look at
different places, not just a reordering of the same nearby ones. A place found by one of these
searches gets an `intent_match` bonus.

**Best rated.** Only open places rated 4.0+ with 30+ reviews are considered. Ratings are
adjusted by review count with a Bayesian average, `(n·R + 100·3.9) / (n + 100)`, so a 4.9 from 5
reviews doesn't beat a 4.6 from 3,000. Since everything shown is already well rated, rating
only breaks ties (weight 1.0). Mood and cravings decide.

**Tags** (`comfort_food`, `quick_service`, `lively`, `quiet`, `dessert`, `familiar`, `outdoor`,
`spicy`, `light`, `coffee`) are scored 0–1 from Google's attributes (live music, cocktails,
outdoor seating, …), the place type, whole-word keywords in the name and Google's description.
Strong evidence = 1.0. Weak evidence such as "serves dessert" (true of most restaurants)
= 0.3–0.5. Only strong tags are shown on cards. *Pure veg* comes from vegetarian place types,
names like "Pure Veg" or "Udupi", and descriptions, and is never set if the name mentions meat.

**Rules.** Mood conditions (e.g. `arousal_below: -0.3`) and cravings map to feature weights.
Score = base weights (rating 1.0, closeness 0.5, intent match 0.8) + every rule that fires +
every craving chosen. You can tune it without touching code.

**LinUCB.** The context is `outer(mood, restaurant).ravel()` plus a one-hot time of day,
88 dimensions in total:
- mood (6): bias, valence, arousal, stress, sad, happy
- restaurant (14): bias, rating, closeness, intent match and the 10 tags
- time of day (4)

Reward is 1 for "Going" and 0 for "Not for me", scoring θᵀx + α·√(xᵀA⁻¹x). Once there are 20
feedback events, this learned score is **added** to the rule score instead of replacing it, so
cravings and diet keep working while it learns what you actually pick. It is refit at startup and
on **Retrain**.

**Variety.** The top 5 are chosen greedily with at most 2 per cuisine, and each repeat of a
cuisine costs a small penalty.

**Measuring accuracy.** Settings shows the *Going rate*, the share of suggestions you marked
Going, overall and by ranking type (rules vs learned). That's the number to watch. It only
accumulates on hosting if `DATABASE_URL` is set.

## Privacy

- **Consent first.** The camera only appears after an explicit checkbox, with a plain
  explanation. You can always pick a mood instead.
- **No images stored.** The photo is decoded in memory, analysed and dropped. It is never written
  to disk or the database, and the schema has no image column (a test checks this). Only 9
  numbers are kept.
- **Only you are analysed.** If more than one face is detected, nothing is analysed.
- **Location.** It is used only for the search and sent only to Google Places. The database stores
  each place's distance, never where you were.
- **Data minimisation and erasure (DPDP-friendly).** No account, name or device ID.
  *Forget my mood* and *Delete my feedback* are in Settings.

This is a prototype, not legal advice. A production deployment under India's DPDP Act would also
need a published privacy notice, a consent-withdrawal flow, retention limits and a grievance contact.

## Accuracy and limitations

- **Location:** with GPS on, a phone is usually accurate to about 5–20 m outdoors. Laptops
  often guess from the network, and the app warns when the fix is poor or far from Hyderabad.
- **Mood:** the emotion model is right about 6 times in 10 on its benchmark (AffectNet), and an
  expression isn't the same as a feeling. That is why the one-tap correction exists, and why
  corrections are what the learner trains on.
- **Restaurants:** ratings, review counts and open-now status come from Google and are only as
  current as Google's data. Demo mode is made up.
- Tags are heuristics based on place types and names.
