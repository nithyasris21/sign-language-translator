"""Sliding-window inference over a stream of landmark frames."""
import json
import time
from collections import deque
from threading import Lock

import numpy as np

from config import (
    CONFIDENCE_THRESHOLD,
    FEATURE_SIZE,
    LABELS_PATH,
    MODEL_PATH,
    SEQUENCE_LENGTH,
    WINDOW_SECONDS,
)


class SignRecognizer:
    """Loads the trained model once; each client gets its own frame buffer."""

    def __init__(self):
        self._model = None
        self._labels: list[str] = []
        self._lock = Lock()

    @property
    def ready(self) -> bool:
        return MODEL_PATH.exists() and LABELS_PATH.exists()

    def load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.ready:
                raise RuntimeError("No trained model. Run `python -m ml.train` first.")
            from tensorflow import keras  # imported lazily: TF startup is slow

            self._model = keras.models.load_model(MODEL_PATH)
            self._labels = json.loads(LABELS_PATH.read_text())

    @property
    def labels(self) -> list[str]:
        if not self._labels and self.ready:
            self._labels = json.loads(LABELS_PATH.read_text())
        return self._labels

    def predict(self, window: np.ndarray) -> tuple[str, float, dict]:
        self.load()
        # predict_on_batch skips predict()'s per-call dataset/callback setup,
        # which dominates for a single window
        probs = np.asarray(self._model.predict_on_batch(window[None, ...]))[0]
        idx = int(np.argmax(probs))
        scores = {label: float(p) for label, p in zip(self._labels, probs)}
        return self._labels[idx], float(probs[idx]), scores


class FrameBuffer:
    """Per-connection rolling window of timestamped landmark vectors.

    Training sequences are SEQUENCE_LENGTH frames spread over a whole clip
    (~2 s), but live frames arrive at whatever rate the CPU can run Holistic
    (~4-6 fps). Taking the last N frames would make the window's time span
    depend on machine speed, so instead the last WINDOW_SECONDS are resampled
    to SEQUENCE_LENGTH slots, the same way preprocess.resample does offline.
    """

    MIN_FRAMES = 5

    def __init__(self, window_seconds: float = WINDOW_SECONDS):
        self.window = window_seconds
        self.frames: deque[tuple[float, np.ndarray]] = deque()
        self.last_emitted: str | None = None
        self._low_since: float | None = None

    @property
    def filled(self) -> int:
        """Window coverage in SEQUENCE_LENGTH units, for the progress bar."""
        if not self.frames:
            return 0
        span = self.frames[-1][0] - self.frames[0][0]
        return min(SEQUENCE_LENGTH, round(SEQUENCE_LENGTH * span / self.window))

    def push(self, keypoints: np.ndarray, t: float | None = None) -> np.ndarray | None:
        t = time.monotonic() if t is None else t
        self.frames.append((t, keypoints))
        # keep one frame at or before the window start so the first slot has data
        while len(self.frames) > 1 and self.frames[1][0] <= t - self.window:
            self.frames.popleft()
        if len(self.frames) < self.MIN_FRAMES or t - self.frames[0][0] < self.window * 0.95:
            return None
        times = np.array([ft for ft, _ in self.frames])
        targets = np.linspace(t - self.window, t, SEQUENCE_LENGTH)
        idx = np.clip(np.searchsorted(times, targets, side="right") - 1, 0, len(times) - 1)
        return np.array([self.frames[i][1] for i in idx], dtype=np.float32)

    def reset(self):
        self.frames.clear()
        self.last_emitted = None
        self._low_since = None

    def accept(self, label: str, confidence: float, t: float | None = None) -> bool:
        """Debounce: only surface a new word above threshold, not the same one
        repeated on every frame of the same gesture."""
        if t is None:
            t = self.frames[-1][0] if self.frames else time.monotonic()
        if confidence < CONFIDENCE_THRESHOLD:
            # the same word only counts as new once confidence has stayed low
            # for a whole window: until then the window still overlaps the
            # previous gesture and can re-fire on it ("thin thin" for one sign)
            if self._low_since is None:
                self._low_since = t
            if t - self._low_since >= self.window:
                self.last_emitted = None
            return False
        self._low_since = None
        if label == self.last_emitted:
            return False
        self.last_emitted = label
        return True


recognizer = SignRecognizer()
