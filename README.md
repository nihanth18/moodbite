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
   - **Secrets:** paste `GOOGLE_PLACES_API_KEY = "your real key"` (see `.streamlit/secrets.toml.example`)
4. Click **Deploy**. The first build takes a few minutes, because MediaPipe and ONNX are large.
   `packages.txt` installs the system libraries OpenCV needs.
5. **Lock it down.** In the app's **Settings → Sharing**, restrict viewers to specific email
   addresses. Anyone with a public link can otherwise run searches on your Google bill.
6. **Protect the key.** In Google Cloud → Credentials, restrict the key to *Places API (New)*
   and set a daily quota or budget alert. Each search makes 2 Places calls (nearby + top-rated).

Things to know about Community Cloud:
- **Feedback resets on restart.** The server's disk is temporary, so the SQLite feedback log
  is wiped when the app restarts or redeploys. To keep learning long-term, move `storage.py` to a
  hosted database (e.g. Postgres) later.
- **The app sleeps** after a period without visitors. The first visit after that takes about a minute to wake it.
- **Photos are processed on Streamlit's servers** (in memory, never saved), not on your PC.
  The in-app privacy note says "on the MoodBite server" for this reason.

Other options: any Docker host (Render, Railway, Fly.io, Google Cloud Run) works with
`streamlit run app.py --server.port $PORT --server.address 0.0.0.0` and the same packages,
plus `GOOGLE_PLACES_API_KEY` as an environment variable.

## Using the app

1. **How are you feeling?** Take a selfie (after ticking consent) or tap a mood. After a
   selfie you can correct the detected mood in one tap.
2. **Where are you?** *Near me* uses your phone's GPS (high-accuracy fix, error margin shown,
   *Refresh* button), or search an area by name. Set how far you'll go and a minimum rating
   (4.0★, 4.3★ or 4.5★).
3. **Your top picks.** Five open, top-rated places, each with rating, review count, distance,
   walk or drive time, a one-line reason, mood tags, a **Directions** button (Google Maps), and
   **Going here** / **Not for me** feedback.

Settings (demo mode, retraining, deleting your data) are at the bottom of the page.

## How it works

```
photo ─► MediaPipe face detection ─► crop ─► HSEmotion (enet_b0_8_va_mtl, + mirrored copy)
          │ 0 faces / >1 face / low score → friendly message, nothing analysed
          ▼
     mood vector: 7 emotion probs + valence + arousal   (image discarded here)
          ▼
Places: searchNearby + searchText(openNow, minRating) ─► merge ─► filter: open now,
        rating ≥ 4.0 (or your choice), ≥ 30 reviews, within distance ─► feature tagging
          ▼
ranker: rules (<20 feedback events) or LinUCB (≥20) ─► top 5 + one-line "why"
          ▼
"Going here" / "Not for me" ─► SQLite (numbers + restaurant features only)
```

| File | What it does |
|---|---|
| `app.py` | Streamlit UI (mobile-first; theme in `.streamlit/config.toml`). |
| `emotion.py` | MediaPipe face detection (Tasks API) and HSEmotion emotion, valence and arousal. Handles no face, multiple faces (refused: only the consenting user is analysed), low detection confidence, tiny faces, and uncertain expressions (used at half weight). |
| `places.py` | Places API (New) client, best-rated filtering, feature tagging, demo data. |
| `ranker.py` | Rule ranker driven by `config/mood_rules.yaml`, plus LinUCB and the `Recommender` that switches between them. |
| `storage.py` | SQLite feedback log, with "delete all my data". |
| `train.py` | Offline baseline: SGD logistic regression, prints accuracy and AUC against a majority-class baseline. |

**Best rated.** Ratings are adjusted by review count with a Bayesian average,
`(n·R + 100·3.9) / (n + 100)`, so a 4.9★ from 5 reviews doesn't beat a 4.6★ from 3,000.
Places with fewer than 30 reviews are dropped. Rating has the largest weight in the ranking, so
your mood decides between places that are all highly rated. There is no price filter.

**Tags** are inferred from place types and whole-word name keywords: `comfort_food`,
`quick_service`, `lively`, `quiet`, `dessert`, `familiar` (a known chain or 1000+ reviews).

**Phase 1, rules.** `config/mood_rules.yaml` maps mood conditions (e.g. `arousal_below: -0.1`)
to tag weights (e.g. `comfort_food: 1.2`). Score = base weights (rating 2.0, closeness 0.6) + the
weights of every rule that fires. You can tune it without touching code.

**Phase 2, LinUCB.** The context is `outer(mood, restaurant).ravel()` plus a one-hot time of day,
58 dimensions in total:
- mood (6): bias, valence, arousal, stress, sad, happy
- restaurant (9): bias, rating, closeness and the 6 tags
- time of day (4)

It is a shared-parameter LinUCB with reward 1 for "Going here" and 0 for "Not for me", scoring
θᵀx + α·√(xᵀA⁻¹x). It is refit from SQLite at startup and on **Retrain**. Below 20 events
the app uses the rules.

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
