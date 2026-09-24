"""Record training sequences for one sign from the webcam.

    python -m ml.collect hello
    python -m ml.collect thanks --count 60

Each run writes SEQUENCES_PER_SIGN samples of SEQUENCE_LENGTH frames to
data/sequences/<sign>/<n>.npy. Press 'q' to abort, any key to skip the
countdown. Vary distance, angle and speed between takes -- the model
generalises only as far as your recordings do.
"""
import argparse
import sys

import cv2
import numpy as np

from config import SEQUENCES_PER_SIGN, SEQUENCE_LENGTH, SEQ_DIR
from ml.landmarks import detect, draw, extract_keypoints, make_holistic


def next_index(sign_dir) -> int:
    existing = sorted(int(p.stem) for p in sign_dir.glob("*.npy") if p.stem.isdigit())
    return existing[-1] + 1 if existing else 0


def banner(frame, text, sub=""):
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    if sub:
        cv2.putText(frame, sub, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sign", help="word/label to record, e.g. hello")
    ap.add_argument("--count", type=int, default=SEQUENCES_PER_SIGN)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    sign_dir = SEQ_DIR / args.sign
    sign_dir.mkdir(parents=True, exist_ok=True)
    start = next_index(sign_dir)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")

    with make_holistic() as holistic:
        for take in range(start, start + args.count):
            # countdown so you can get into position
            for _ in range(30):
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                results = detect(frame, holistic)
                draw(frame, results)
                banner(frame, f"GET READY: {args.sign}", f"take {take}  (q to quit)")
                cv2.imshow("collect", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return

            window = []
            while len(window) < SEQUENCE_LENGTH:
                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                results = detect(frame, holistic)
                window.append(extract_keypoints(results))
                draw(frame, results)
                banner(frame, f"RECORDING {args.sign}", f"frame {len(window)}/{SEQUENCE_LENGTH}")
                cv2.imshow("collect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release(); cv2.destroyAllWindows(); return

            np.save(sign_dir / f"{take}.npy", np.array(window, dtype=np.float32))
            print(f"saved {args.sign}/{take}.npy")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
