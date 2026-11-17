import numpy as np
import pytest

from emotion import EMOTIONS, EmotionAnalyzer, MoodVector, crop_face, map_hsemotion

IMG = np.zeros((240, 320, 3), dtype=np.uint8)


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, rgb):
        return self.faces


class FakeRecognizer:
    def __init__(self, probs=None, va=(0.5, 0.2), fail=False):
        self.probs = probs or {**{e: 0.0 for e in EMOTIONS}, "happy": 0.9, "neutral": 0.1}
        self.va, self.fail, self.calls = va, fail, 0

    def predict(self, face):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model missing")
        return self.probs, *self.va


def analyzer(faces, **kw):
    rec = kw.pop("recognizer", FakeRecognizer())
    return EmotionAnalyzer(0.6, 0.3, detector=FakeDetector(faces), recognizer=rec), rec


def test_ok():
    a, _ = analyzer([(100, 60, 80, 100, 0.95)])
    res = a.analyze(IMG)
    assert res.status == "ok" and res.usable
    assert res.mood.top_emotion == "happy"
    assert res.mood.valence == 0.5 and res.mood.arousal == 0.2


def test_no_face():
    a, rec = analyzer([])
    res = a.analyze(IMG)
    assert res.status == "no_face" and not res.usable and rec.calls == 0


def test_multiple_faces_are_not_analysed():
    a, rec = analyzer([(10, 10, 60, 60, 0.9), (150, 10, 60, 60, 0.85)])
    res = a.analyze(IMG)
    assert res.status == "multiple_faces" and res.n_faces == 2
    assert rec.calls == 0  # nobody's expression is analysed without consent


def test_one_confident_face_plus_weak_detection_is_ok():
    a, _ = analyzer([(100, 60, 80, 100, 0.95), (5, 5, 20, 20, 0.35)])
    assert a.analyze(IMG).status == "ok"


def test_low_confidence_detection():
    a, rec = analyzer([(100, 60, 80, 100, 0.4)])
    res = a.analyze(IMG)
    assert res.status == "low_confidence" and "40%" in res.message and rec.calls == 0


def test_tiny_face():
    a, _ = analyzer([(100, 60, 10, 10, 0.9)])
    assert a.analyze(IMG).status == "low_confidence"


def test_uncertain_emotion_still_returns_mood():
    flat = {e: 1 / 7 for e in EMOTIONS}
    a, _ = analyzer([(100, 60, 80, 100, 0.9)], recognizer=FakeRecognizer(flat))
    res = a.analyze(IMG)
    assert res.status == "uncertain" and res.usable


def test_model_failure_is_graceful():
    a, _ = analyzer([(100, 60, 80, 100, 0.9)], recognizer=FakeRecognizer(fail=True))
    res = a.analyze(IMG)
    assert res.status == "error" and "model missing" in res.message


def test_bad_image():
    a, _ = analyzer([])
    assert a.analyze(np.zeros((10, 10, 3), dtype=np.uint8)).status == "error"


def test_map_hsemotion_folds_contempt_into_disgust():
    labels = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness",
              "Surprise"]
    p = map_hsemotion(labels, np.zeros(8))
    assert set(p) == set(EMOTIONS)
    assert p["disgust"] == pytest.approx(2 / 8)
    assert sum(p.values()) == pytest.approx(1.0)


def test_mood_vector_roundtrip_is_numeric_only():
    m = MoodVector({**{e: 0.0 for e in EMOTIONS}, "angry": 0.3, "fear": 0.2, "neutral": 0.5},
                   -0.2, 0.4)
    d = m.to_dict()
    assert len(d) == 9 and all(isinstance(v, float) for v in d.values())
    assert MoodVector.from_dict(d).to_dict() == d
    assert m.stress == pytest.approx(0.5)


def test_crop_face_clips_to_image():
    # box spans -20..80, +15px margin each side -> -35..95, clipped to 0..95
    assert crop_face(IMG, (-20, -20, 100, 100)).shape[:2] == (95, 95)
    assert crop_face(IMG, (300, 200, 100, 100)).shape[:2] == (55, 35)  # clipped at right/bottom edge
