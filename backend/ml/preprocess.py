"""Turn downloaded WLASL videos into LSTM-ready landmark sequences.

    python -m ml.preprocess

Reads data/raw/<gloss>/*.mp4, writes data/sequences/<gloss>/*.npy in exactly
the format ml/collect.py produces, so webcam-recorded and dataset-derived
samples can be mixed freely.

WLASL gives ~14 clips per gloss, far too few for an LSTM, so each clip is
expanded two ways:

  * mirroring      - a left-handed signing of the same sign. MediaPipe is run
                     again on the flipped frame rather than negating x in the
                     feature vector, because pose landmarks are indexed by
                     body side and a naive flip would swap their meaning.
  * temporal jitter - the clip has ~74 frames and we need 30, so each take
                     resamples with a different offset and stride, which also
                     teaches the model tolerance to signing speed.
"""
import argparse
import os
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

from config import DATA_DIR, FEATURE_SIZE, SEQUENCE_LENGTH, SEQ_DIR
from ml.landmarks import detect, extract_keypoints, make_holistic

RAW_DIR = DATA_DIR / "raw"

# MediaPipe Holistic manages ~1.5 fps per core here, so a few hundred clips is
# hours single-threaded. Each worker process builds its own Holistic instance
# (the object is neither picklable nor thread-safe) and handles whole clips.
_HOLISTIC = None


def _worker_holistic():
    global _HOLISTIC
    if _HOLISTIC is None:
        _HOLISTIC = make_holistic()
    return _HOLISTIC


def process_clip(job):
    """Runs in a worker process: one clip -> its .npy files on disk."""
    clip_str, out_str, takes, mirrors, min_hand = job
    clip_path, out_dir = pathlib.Path(clip_str), pathlib.Path(out_str)
    holistic = _worker_holistic()
    written, dropped = 0, 0
    for mirror in mirrors:
        track = landmark_track(clip_path, holistic, mirror)
        if hand_fraction(track) < min_hand:
            dropped += 1
            continue
        for take in range(takes):
            seq = resample(track, take, takes)
            if seq is None or seq.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                continue
            np.save(out_dir / f"{clip_path.stem}_{'m' if mirror else 'o'}{take}.npy", seq)
            written += 1
    return clip_path.parent.name, written, dropped


def landmark_track(path, holistic, mirror: bool) -> np.ndarray:
    """Every frame of one video as a (n_frames, FEATURE_SIZE) array."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        frames.append(extract_keypoints(detect(frame, holistic)))
    cap.release()
    return np.array(frames, dtype=np.float32) if frames else np.empty((0, FEATURE_SIZE), np.float32)


def resample(track: np.ndarray, take: int, takes: int) -> np.ndarray | None:
    """Pick SEQUENCE_LENGTH frames spanning the clip, offset per take."""
    n = len(track)
    if n < SEQUENCE_LENGTH // 2:
        return None
    if n <= SEQUENCE_LENGTH:
        # short clip: pad by repeating the final frame
        pad = np.repeat(track[-1:], SEQUENCE_LENGTH - n, axis=0)
        return np.concatenate([track, pad])

    # spread takes across the clip's slack so each sees a different window
    slack = n - SEQUENCE_LENGTH
    start = int(slack * take / max(takes - 1, 1)) if takes > 1 else slack // 2
    idx = np.linspace(start, n - 1, SEQUENCE_LENGTH).astype(int)
    return track[idx]


def hand_fraction(track: np.ndarray) -> float:
    """Share of frames where at least one hand was found. Clips where
    MediaPipe never locks onto a hand carry no usable signal."""
    hands = track[:, 132:]
    return float((np.abs(hands).sum(axis=1) > 0).mean()) if len(track) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes", type=int, default=4, help="temporal crops per clip")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--min-hand-fraction", type=float, default=0.3)
    ap.add_argument("--fresh", action="store_true", help="wipe existing sequences first")
    ap.add_argument("--workers", type=int, default=0, help="0 = cpu_count - 2")
    args = ap.parse_args()

    if not RAW_DIR.exists():
        raise SystemExit("No data/raw -- run `python -m ml.fetch_wlasl` first.")
    if args.fresh and SEQ_DIR.exists():
        shutil.rmtree(SEQ_DIR)
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    glosses = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    mirrors = [False] if args.no_mirror else [False, True]

    jobs = []
    for gloss_dir in glosses:
        out_dir = SEQ_DIR / gloss_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for clip in sorted(gloss_dir.glob("*.mp4")):
            jobs.append((str(clip), str(out_dir), args.takes, mirrors, args.min_hand_fraction))

    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"{len(jobs)} clips over {workers} workers", flush=True)

    written = skipped = 0
    per_gloss = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (gloss, n, dropped) in enumerate(pool.map(process_clip, jobs), 1):
            written += n
            skipped += dropped
            per_gloss[gloss] = per_gloss.get(gloss, 0) + n
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} clips, {written} sequences", flush=True)

    print()
    for gloss in sorted(per_gloss):
        print(f"  {gloss:<14} {per_gloss[gloss]:>4} sequences")

    print(f"\n{written} sequences written, {skipped} clip-passes dropped (no hands detected)")
    print("next: python -m ml.train")


if __name__ == "__main__":
    main()
