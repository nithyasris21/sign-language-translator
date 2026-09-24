"""Record a reference clip for the text -> sign direction.

    python -m ml.record_clip hello --seconds 3

Writes data/signs/words/<word>.mp4, which the frontend plays back when the
user types that word. For fingerspelling, drop a-z images into
data/signs/letters/ instead.
"""
import argparse
import sys
import time

import cv2

from app.library import WORDS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("word")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for n in (3, 2, 1):
        deadline = time.time() + 1
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, str(n), (width // 2 - 20, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            cv2.imshow("record", frame)
            cv2.waitKey(1)

    out_path = WORDS_DIR / f"{args.word.lower()}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (width, height))

    deadline = time.time() + args.seconds
    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        writer.write(frame)
        cv2.putText(frame, "REC", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("record", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
