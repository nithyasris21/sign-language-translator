"""Score Sign -> Text on the text -> sign reference videos in data/signs/words.

    python -m tests.signs_words_test islr
    python -m tests.signs_words_test lstm

Each video whose word is in the model's vocabulary is fed through the live
path (Holistic at LIVE_FPS, the model's time window) and scored by the most
confident window, like a user watching the translation box.

For the lstm model these videos were copied from its own WLASL training
clips, so its score here is optimistic; islr never saw them.
"""
import json
import sys
from pathlib import Path

import cv2

from config import ISLR_LABELS_PATH, SIGN_DIR
from ml.landmarks import detect, extract_keypoints, extract_raw543, make_holistic
from ml.predict import ISLR_DISPLAY, FrameBuffer, ISLRRecognizer, SignRecognizer

LIVE_FPS = 5.0


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else "islr"
    if kind == "islr":
        rec, extract = ISLRRecognizer(), extract_raw543
        vocab = {ISLR_DISPLAY.get(g, g): g for g in json.loads(ISLR_LABELS_PATH.read_text())}
    else:
        rec, extract = SignRecognizer(), extract_keypoints
        vocab = {w: w for w in rec.labels}
    rec.load()

    clips = [(label, SIGN_DIR / "words" / f"{gloss}.mp4") for label, gloss in vocab.items()]
    clips = [(label, p) for label, p in clips if p.exists()]
    top1 = top5 = n = 0
    wrong = []
    for label, path in sorted(clips):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        holistic = make_holistic()
        buf = FrameBuffer(rec.window_seconds, rec.sequence_length)
        best, i, next_t = None, 0, 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t, i = i / fps, i + 1
            if t < next_t:
                continue
            next_t = t + 1.0 / LIVE_FPS
            window = buf.push(extract(detect(frame, holistic)), t)
            if window is not None:
                guess = rec.predict(window)
                if best is None or guess[1] > best[1]:
                    best = guess
        holistic.close()
        if best is None:  # clip shorter than the window
            continue
        n += 1
        ranked = sorted(best[2], key=best[2].get, reverse=True)
        top1 += ranked[0] == label
        top5 += label in ranked[:5]
        if ranked[0] != label:
            wrong.append(f"{label}->{ranked[0]}")
        print(f"{label:<14} -> {best[0]:<14} {best[1]:.2f}", flush=True)

    print(f"\n[{kind}] {n} of {len(vocab)} vocabulary words have a signs/words video")
    print(f"top-1 {top1 / n:.3f}  top-5 {top5 / n:.3f}  chance {1 / len(vocab):.3f}")
    print("missed:", ", ".join(wrong))


if __name__ == "__main__":
    main()
