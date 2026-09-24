"""End-to-end sanity check that does not need a webcam or recorded data.

    python -m tests.smoke_test

Verifies MediaPipe loads and produces a correctly shaped feature vector, that
the LSTM trains and overfits synthetic sequences, and that the text -> sign
lookup and FastAPI routes respond. Run this before spending an hour recording
takes, so a broken install surfaces early.
"""
import sys

import numpy as np

from config import FEATURE_SIZE, SEQUENCE_LENGTH

FAILURES = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    except Exception as exc:  # noqa: BLE001 - smoke test reports, never raises
        FAILURES.append(name)
        print(f"  FAIL  {name} — {type(exc).__name__}: {exc}")


def test_landmarks():
    from ml.landmarks import detect, extract_keypoints, make_holistic

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    with make_holistic(static=True) as holistic:
        results = detect(blank, holistic)
        vec = extract_keypoints(results)
    assert vec.shape == (FEATURE_SIZE,), f"got {vec.shape}, expected ({FEATURE_SIZE},)"
    return f"feature vector {vec.shape}"


def test_training():
    """Two synthetic classes with distinct signatures; the model must separate
    them. If this cannot overfit 40 easy samples, the architecture is broken."""
    from ml.train import build_model

    rng = np.random.default_rng(0)
    n = 20
    a = rng.normal(0.0, 0.05, (n, SEQUENCE_LENGTH, FEATURE_SIZE))
    b = rng.normal(0.0, 0.05, (n, SEQUENCE_LENGTH, FEATURE_SIZE))
    b[:, :, :40] += 1.0                       # class B has a strong offset
    X = np.concatenate([a, b]).astype(np.float32)
    y = np.array([0] * n + [1] * n)

    model = build_model(2)
    model.fit(X, y, epochs=40, batch_size=8, verbose=0)
    acc = model.evaluate(X, y, verbose=0)[1]
    assert acc > 0.9, f"only reached {acc:.2f} on separable synthetic data"
    return f"train accuracy {acc:.2f}"


def test_predict_buffer():
    from ml.predict import FrameBuffer

    buf = FrameBuffer(window_seconds=2.0)
    for i in range(SEQUENCE_LENGTH - 1):
        assert buf.push(np.zeros(FEATURE_SIZE, dtype=np.float32), t=i / 15) is None, "filled too early"
    window = buf.push(np.zeros(FEATURE_SIZE, dtype=np.float32), t=(SEQUENCE_LENGTH - 1) / 15)
    assert window is not None and window.shape == (SEQUENCE_LENGTH, FEATURE_SIZE)

    # a slow server (4 fps) must still yield a 2 s window, in time order
    slow = FrameBuffer(window_seconds=2.0)
    for i in range(12):
        window = slow.push(np.full(FEATURE_SIZE, i, dtype=np.float32), t=i / 4)
    assert window is not None and window.shape == (SEQUENCE_LENGTH, FEATURE_SIZE)
    firsts = window[:, 0]
    assert firsts[0] == 3 and firsts[-1] == 11 and np.all(np.diff(firsts) >= 0), firsts

    assert buf.accept("hello", 0.9, t=10.0) is True, "first confident hit should emit"
    assert buf.accept("hello", 0.9, t=10.1) is False, "repeat of same label should be suppressed"
    assert buf.accept("hello", 0.1, t=10.2) is False, "below threshold should not emit"
    assert buf.accept("hello", 0.9, t=10.3) is False, "one low frame mid-gesture is not a new word"
    buf.accept("hello", 0.1, t=11.0)
    buf.accept("hello", 0.1, t=12.0)
    assert buf.accept("hello", 0.9, t=12.1) is False, "window still overlaps the last gesture"
    buf.accept("hello", 0.1, t=13.0)
    buf.accept("hello", 0.1, t=15.0)
    assert buf.accept("hello", 0.9, t=15.1) is True, "should re-emit after gesture ended"
    return "window fills and debounce behaves"


def test_library():
    from app import library

    assert library.tokenize("Hello, world!") == ["hello", "world"]
    seq = library.translate("zzz")
    assert seq and seq[0]["label"] == "zzz"
    return f"{len(library.available_words())} word clip(s) installed"


def test_api():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200, health.status_code
    assert "model_trained" in health.json()

    resp = client.post("/api/text-to-sign", json={"text": "hello world"})
    assert resp.status_code == 200, resp.status_code
    assert "sequence" in resp.json()
    return f"model_trained={health.json()['model_trained']}"


def main():
    print("smoke test\n")
    check("mediapipe landmark extraction", test_landmarks)
    check("lstm trains on synthetic data", test_training)
    check("sliding window + debounce", test_predict_buffer)
    check("text -> sign lookup", test_library)
    check("fastapi routes", test_api)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed — ready to record training data")


if __name__ == "__main__":
    main()
