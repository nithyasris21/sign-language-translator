"""One-off: grow the 146-word set to 200 words and retrain.

Downloads WLASL clips for extra candidate words, keeps the ones with the most
working clips, landmarks only those, then trains on all 200. Writes
logs/DONE (results) or logs/FAILED.
"""
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

TARGET = 200
PY = sys.executable
LOGS = Path("logs")
RAW, PARKED = Path("data/raw"), Path("data/raw_parked")


def run(args, log):
    with open(LOGS / log, "w", encoding="utf-8") as f:
        subprocess.run([PY, "-m", *args], stdout=f, stderr=subprocess.STDOUT, check=True,
                       env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})


def main():
    for name in ("DONE", "FAILED"):
        (LOGS / name).unlink(missing_ok=True)
    have = set(json.loads(Path("data/models/labels_146words.json").read_text()))
    extra = Path("logs/words_extra.txt").read_text().split(" ")
    extra = [w for w in " ".join(extra).replace("come here", "come_here").replace("a lot", "a_lot").split()]
    extra = [w.replace("_", " ") for w in extra]

    run(["ml.fetch_wlasl", "--words", *extra], "fetch_200.log")

    counts = {w: len(list((RAW / w).glob("*.mp4"))) for w in extra if (RAW / w).exists()}
    chosen = sorted((w for w in counts if counts[w] >= 4), key=counts.get, reverse=True)
    chosen = chosen[: TARGET - len(have)]
    (LOGS / "words_200.txt").write_text("\n".join(sorted(have | set(chosen))))

    # landmark only the newly chosen words; everything else waits in PARKED
    PARKED.mkdir(exist_ok=True)
    for d in list(RAW.iterdir()):
        if d.name not in chosen:
            shutil.move(str(d), str(PARKED / d.name))
    run(["ml.preprocess"], "preprocess_200.log")
    for d in list(PARKED.iterdir()):
        if d.name in have:
            shutil.move(str(d), str(RAW / d.name))

    run(["ml.train"], "train.log")
    lines = [l for l in (LOGS / "train.log").read_text(encoding="utf-8", errors="replace").splitlines()
             if "accuracy:" in l and "held-out" in l or "baseline" in l or "samples across" in l]
    (LOGS / "DONE").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (LOGS / "FAILED").write_text(traceback.format_exc(), encoding="utf-8")
        raise
