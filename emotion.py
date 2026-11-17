"""Face detection (MediaPipe) + emotion / valence / arousal (HSEmotion ONNX).

Privacy: images are decoded in memory only. Nothing here writes image data to disk; the only
output is a small numeric MoodVector.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

EMOTIONS = ("happy", "sad", "angry", "neutral", "surprise", "fear", "disgust")

# HSEmotion 8-class labels -> our 7 (contempt is folded into disgust)
_HSE_MAP = {
    "anger": "angry", "contempt": "disgust", "disgust": "disgust", "fear": "fear",
    "happiness": "happy", "neutral": "neutral", "sadness": "sad", "surprise": "surprise",
}
HSE_MODEL = "enet_b0_8_va_mtl"  # multi-task model: 8 emotions + valence + arousal
MP_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


@dataclass
class MoodVector:
    probs: dict[str, float]
    valence: float  # -1 (negative) .. 1 (positive)
    arousal: float  # -1 (calm) .. 1 (excited)
    face_confidence: float = 1.0

    @property
    def top_emotion(self) -> str:
        return max(self.probs, key=self.probs.get)

    @property
    def confidence(self) -> float:
        return float(max(self.probs.values()))

    @property
    def stress(self) -> float:
        return float(self.probs.get("angry", 0) + self.probs.get("fear", 0)
                     + self.probs.get("disgust", 0))

    def to_dict(self) -> dict[str, float]:
        """Numeric-only representation (what gets stored)."""
        return {**{e: float(self.probs.get(e, 0.0)) for e in EMOTIONS},
                "valence": float(self.valence), "arousal": float(self.arousal)}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "MoodVector":
        return cls(probs={e: float(d.get(e, 0.0)) for e in EMOTIONS},
                   valence=float(d.get("valence", 0.0)), arousal=float(d.get("arousal", 0.0)))


@dataclass
class AnalysisResult:
    status: str  # ok | uncertain | no_face | multiple_faces | low_confidence | error
    message: str
    mood: MoodVector | None = None
    n_faces: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.mood is not None


class FaceDetector(Protocol):
    def detect(self, rgb: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """Return [(x, y, w, h, score)] in pixel coordinates."""


class EmotionRecognizer(Protocol):
    def predict(self, face_rgb: np.ndarray) -> tuple[dict[str, float], float, float]:
        """Return (probs over EMOTIONS, valence, arousal)."""


# --- MediaPipe ------------------------------------------------------------------------------
class MediaPipeDetector:
    """Uses the legacy `mp.solutions` API when present, else the Tasks API."""

    def __init__(self, min_score: float = 0.3, model_dir: str | Path = "models"):
        import mediapipe as mp

        self._mp = mp
        self._legacy = None
        self._task = None
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "face_detection"):
            self._legacy = solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=min_score)
        else:
            from mediapipe.tasks.python import BaseOptions, vision

            model_path = Path(model_dir) / "blaze_face_short_range.tflite"
            if not model_path.exists():
                import requests

                model_path.parent.mkdir(parents=True, exist_ok=True)
                resp = requests.get(MP_TASK_MODEL_URL, timeout=30)
                resp.raise_for_status()
                model_path.write_bytes(resp.content)  # model weights, not user data
            self._task = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                min_detection_confidence=min_score))

    def detect(self, rgb: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        h, w = rgb.shape[:2]
        faces = []
        if self._legacy is not None:
            res = self._legacy.process(rgb)
            for d in res.detections or []:
                bb = d.location_data.relative_bounding_box
                faces.append((int(bb.xmin * w), int(bb.ymin * h), int(bb.width * w),
                              int(bb.height * h), float(d.score[0])))
        else:
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(rgb))
            for d in self._task.detect(img).detections:
                bb = d.bounding_box
                faces.append((bb.origin_x, bb.origin_y, bb.width, bb.height,
                              float(d.categories[0].score)))
        return faces


# --- HSEmotion ------------------------------------------------------------------------------
class HSEmotionModel:
    def __init__(self, model_name: str = HSE_MODEL):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        self._rec = HSEmotionRecognizer(model_name=model_name)
        self._labels = [self._rec.idx_to_class[i] for i in range(len(self._rec.idx_to_class))]

    def predict(self, face_rgb: np.ndarray) -> tuple[dict[str, float], float, float]:
        # Test-time augmentation: average the original and mirrored face. This reduces
        # sensitivity to head tilt and one-sided lighting.
        views = [np.ascontiguousarray(face_rgb), np.ascontiguousarray(face_rgb[:, ::-1])]
        scores = np.mean([np.asarray(self._rec.predict_emotions(v, logits=True)[1],
                                     dtype=float).ravel() for v in views], axis=0)
        n = len(self._labels)
        logits = scores[:n]
        valence, arousal = (scores[n], scores[n + 1]) if len(scores) >= n + 2 else (0.0, 0.0)
        return map_hsemotion(self._labels, logits), float(np.clip(valence, -1, 1)), \
            float(np.clip(arousal, -1, 1))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def map_hsemotion(labels: list[str], logits: np.ndarray) -> dict[str, float]:
    p = softmax(np.asarray(logits, dtype=float))
    out = {e: 0.0 for e in EMOTIONS}
    for label, prob in zip(labels, p):
        key = _HSE_MAP.get(label.lower())
        if key:
            out[key] += float(prob)
    return out


# --- pipeline -------------------------------------------------------------------------------
def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes to an RGB array in memory (never touches disk)."""
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        # phone photos may carry an EXIF rotation; a sideways face won't be detected
        return np.array(ImageOps.exif_transpose(im).convert("RGB"))


def crop_face(rgb: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.15) -> np.ndarray:
    h, w = rgb.shape[:2]
    x, y, bw, bh = box
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    return rgb[y0:y1, x0:x1]


class EmotionAnalyzer:
    def __init__(
        self,
        min_face_confidence: float = 0.6,
        min_emotion_confidence: float = 0.3,
        detector: FaceDetector | None = None,
        recognizer: EmotionRecognizer | None = None,
    ):
        self.min_face_confidence = min_face_confidence
        self.min_emotion_confidence = min_emotion_confidence
        self._detector = detector
        self._recognizer = recognizer

    @property
    def detector(self) -> FaceDetector:
        if self._detector is None:
            self._detector = MediaPipeDetector()
        return self._detector

    @property
    def recognizer(self) -> EmotionRecognizer:
        if self._recognizer is None:
            self._recognizer = HSEmotionModel()
        return self._recognizer

    def analyze(self, rgb: np.ndarray) -> AnalysisResult:
        if rgb is None or rgb.ndim != 3 or rgb.shape[0] < 32 or rgb.shape[1] < 32:
            return AnalysisResult("error", "Couldn't read that image. Please try again.")

        faces = self.detector.detect(rgb)
        if not faces:
            return AnalysisResult("no_face", "No face detected. Face the camera in good light "
                                  "and try again.")
        confident = [f for f in faces if f[4] >= self.min_face_confidence]
        if len(confident) > 1:
            # Consent is only given by the user, so we don't analyse anyone else in frame.
            return AnalysisResult("multiple_faces", f"{len(confident)} faces detected. Please "
                                  "make sure only you are in frame.", n_faces=len(confident))
        if not confident:
            best = max(f[4] for f in faces)
            return AnalysisResult("low_confidence", f"Face detection confidence is low "
                                  f"({best:.0%}). Try better lighting or move closer.",
                                  n_faces=len(faces))

        x, y, w, h, score = confident[0]
        face = crop_face(rgb, (x, y, w, h))
        if face.size == 0 or min(face.shape[:2]) < 24:
            return AnalysisResult("low_confidence", "The face is too small. Move closer to the "
                                  "camera.", n_faces=1)
        try:
            probs, valence, arousal = self.recognizer.predict(face)
        except Exception as e:  # model download / runtime failures
            return AnalysisResult("error", f"Emotion model failed: {e}", n_faces=1)
        mood = MoodVector(probs=probs, valence=valence, arousal=arousal, face_confidence=score)
        if mood.confidence < self.min_emotion_confidence:
            return AnalysisResult("uncertain", "Your expression is hard to read, so "
                                  "recommendations lean on ratings and distance.",
                                  mood=mood, n_faces=1)
        return AnalysisResult("ok", f"You look mostly {mood.top_emotion}.", mood=mood, n_faces=1)
