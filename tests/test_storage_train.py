import os
import sqlite3
from datetime import datetime

import pytest

from conftest import make_mood
from storage import Storage, is_postgres_url
from train import simulate_events, train_baseline

# Set TEST_DATABASE_URL=postgresql://... to also run every storage test against Postgres.
PG_URL = os.getenv("TEST_DATABASE_URL", "")


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    if request.param == "sqlite":
        yield Storage(tmp_path / "t.db")
        return
    if not PG_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    s = Storage(PG_URL)
    s.clear()
    yield s
    s.clear()


def test_backend_detection(tmp_path):
    assert is_postgres_url("postgresql://u:p@host/db") and is_postgres_url("postgres://x")
    s = Storage(tmp_path / "x.db")
    assert s.kind == "sqlite" and not s.persistent


def test_metrics_going_rate(store):
    m0 = store.metrics()
    assert m0["n"] == 0 and m0["rate"] is None
    for action, policy in [("going", "rules"), ("not_feeling_it", "rules"),
                           ("going", "linucb"), ("going", "linucb")]:
        store.log_feedback({"happy": 1.0}, {}, action, policy)
    m = store.metrics()
    assert m["n"] == 4 and m["going"] == 3 and m["rate"] == 0.75
    assert m["by_policy"]["rules"]["rate"] == 0.5
    assert m["by_policy"]["linucb"] == {"n": 2, "going": 2, "rate": 1.0}


def test_log_and_load(store):
    mood = make_mood(0.2, -0.3, happy=0.6, neutral=0.4).to_dict()
    feats = {"place_id": "p1", "comfort_food": 1, "rating": 4.5}
    store.log_feedback(mood, feats, "going", "rules", datetime(2026, 5, 1, 19, 30))
    store.log_feedback(mood, feats, "not_feeling_it")
    ev = store.load_events()
    assert store.count() == 2
    assert ev[0]["reward"] == 1 and ev[1]["reward"] == 0
    assert ev[0]["hour"] == 19 and ev[0]["mood"] == mood and ev[0]["restaurant"] == feats


def test_rejects_bad_action_and_non_numeric_mood(store):
    with pytest.raises(ValueError):
        store.log_feedback({"happy": 1.0}, {}, "maybe")
    with pytest.raises(ValueError, match="numeric"):
        store.log_feedback({"image": b"jpegbytes"}, {}, "going")


def test_schema_has_no_image_column(store):
    if store.kind == "sqlite":
        cols = [r[1] for r in sqlite3.connect(store.path).execute("PRAGMA table_info(feedback)")]
    else:
        cols = [r["column_name"] for r in store._exec(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'feedback'",
            fetch="all")]
    assert cols and not any("image" in c or "photo" in c or "frame" in c for c in cols)


def test_clear(store):
    store.log_feedback({"happy": 1.0}, {}, "going")
    store.clear()
    assert store.count() == 0


def test_train_baseline_skips_small_or_one_class():
    assert "skipped" in train_baseline(simulate_events(5))
    ev = [dict(e, reward=1) for e in simulate_events(40)]
    assert "skipped" in train_baseline(ev)


def test_train_baseline_metrics_on_simulated_data():
    m = train_baseline(simulate_events(800, seed=3))
    assert m["n_test"] == 200
    assert 0 <= m["accuracy"] <= 1
    assert m["auc"] > 0.6  # simulated user has learnable preferences
