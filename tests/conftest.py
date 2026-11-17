import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from emotion import EMOTIONS, MoodVector  # noqa: E402


def make_mood(valence=0.0, arousal=0.0, **probs) -> MoodVector:
    p = {e: 0.0 for e in EMOTIONS}
    p.update(probs)
    if not any(p.values()):
        p["neutral"] = 1.0
    return MoodVector(p, valence, arousal)


@pytest.fixture
def mood_factory():
    return make_mood
