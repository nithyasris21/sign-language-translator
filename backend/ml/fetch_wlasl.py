"""Download a vocabulary subset of the WLASL word-level ASL corpus.

    python -m ml.fetch_wlasl --top 12
    python -m ml.fetch_wlasl --words hello thanks yes no

WLASL has 2000 glosses but only ~6 clips each (16 at best), so we take the
glosses with the most clips rather than the most useful-sounding ones --
with this little data, sample count decides whether a class is learnable.

Videos land in data/raw/<gloss>/*.mp4 and one representative clip per gloss
is copied to data/signs/words/<gloss>.mp4 for the text -> sign direction.

Source: Voxel51/WLASL on HuggingFace (CC BY-NC 4.0, research use).
"""
import argparse
import collections
import json
import shutil

import requests

from app.library import WORDS_DIR
from config import DATA_DIR

REPO = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/main"
RAW_DIR = DATA_DIR / "raw"
META_PATH = DATA_DIR / "wlasl_samples.json"


def load_metadata() -> list[dict]:
    if not META_PATH.exists():
        print("downloading WLASL metadata (~9 MB)...")
        resp = requests.get(f"{REPO}/samples.json", timeout=180)
        resp.raise_for_status()
        META_PATH.write_bytes(resp.content)
    samples = json.loads(META_PATH.read_text(encoding="utf-8"))["samples"]
    return [s for s in samples if s.get("gloss")]


def pick_glosses(samples, top: int, words: list[str] | None) -> list[str]:
    counts = collections.Counter(s["gloss"]["label"] for s in samples)
    if words:
        missing = [w for w in words if w not in counts]
        if missing:
            print(f"not in WLASL, skipping: {', '.join(missing)}")
        return [w for w in words if w in counts]
    return [g for g, _ in counts.most_common(top)]


def download(sample, dest) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        resp = requests.get(f"{REPO}/{sample['filepath']}", timeout=180)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:  # noqa: BLE001 - a dead clip should not abort the run
        print(f"    failed {sample['filepath']}: {exc}")
        dest.unlink(missing_ok=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12, help="take the N glosses with most clips")
    ap.add_argument("--words", nargs="*", help="explicit gloss list instead of --top")
    ap.add_argument("--max-per-gloss", type=int, default=20)
    args = ap.parse_args()

    samples = load_metadata()
    glosses = pick_glosses(samples, args.top, args.words)
    by_gloss = collections.defaultdict(list)
    for s in samples:
        if s["gloss"]["label"] in glosses:
            by_gloss[s["gloss"]["label"]].append(s)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for gloss in glosses:
        clips = by_gloss[gloss][: args.max_per_gloss]
        out_dir = RAW_DIR / gloss
        out_dir.mkdir(exist_ok=True)
        got = 0
        for s in clips:
            name = s["filepath"].split("/")[-1]
            if download(s, out_dir / name):
                got += 1
        total += got
        print(f"  {gloss:<14} {got}/{len(clips)} clips")

        # longest successful clip doubles as the text -> sign reference
        have = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True)
        if have:
            WORDS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy(have[0], WORDS_DIR / f"{gloss}.mp4")

    print(f"\n{total} clips across {len(glosses)} glosses -> {RAW_DIR}")
    print("next: python -m ml.preprocess")


if __name__ == "__main__":
    main()
