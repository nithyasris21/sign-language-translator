"""Live end-to-end test against a real uvicorn server. No webcam needed.

    python -m tests.live_test

Starts the API on a spare port, then streams the held-out WLASL clips (the
ones train.py never trained on) through /ws/recognize in real time, the way
the browser does: mirrored 480px JPEGs at 12 fps with 1 s of stillness either
side of the sign. Also covers the protocol edge cases the UI rarely hits:
reset, malformed frames, two concurrent clients and an abrupt disconnect.

Needs data/raw (python -m ml.fetch_wlasl) and a trained model. Takes ~3 min.
"""
import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import websockets

from config import BASE_DIR, DATA_DIR

RAW_DIR = DATA_DIR / "raw"
SEND_FPS = 12
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------- helpers

def held_out_clips() -> list[Path]:
    """Same deterministic split as train.py / evaluate.py."""
    from sklearn.model_selection import StratifiedGroupKFold

    from ml.train import load_dataset

    X, y, _, groups = load_dataset()
    _, test_idx = next(StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
                       .split(X, y, groups=groups))
    return [RAW_DIR / f"{g}.mp4" for g in sorted(set(groups[test_idx]))]


def encode_clip(path: Path) -> tuple[list[str], float]:
    """Browser-equivalent frames: mirrored, 480px wide, JPEG q60, padded
    with 1 s of the first/last frame so the sign starts and ends at rest."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        f = cv2.flip(cv2.resize(f, (480, int(h * 480 / w))), 1)
        frames.append("data:image/jpeg;base64,"
                      + base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 60])[1]).decode())
    cap.release()
    pad = int(fps)
    return [frames[0]] * pad + frames + [frames[-1]] * pad, fps


async def stream(url: str, frames: list[str], fps: float) -> dict:
    """Play frames in real time, sampling by wall clock at SEND_FPS."""
    replies = []
    async with websockets.connect(url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        t0 = time.perf_counter()

        async def reader():
            try:
                async for raw in ws:
                    m = json.loads(raw)
                    m["_t"] = time.perf_counter() - t0
                    replies.append(m)
            except websockets.ConnectionClosed:
                pass

        task = asyncio.create_task(reader())
        duration = len(frames) / fps
        while (t := time.perf_counter() - t0) < duration:
            await ws.send(json.dumps({"frame": frames[min(int(t * fps), len(frames) - 1)]}))
            await asyncio.sleep(1 / SEND_FPS)
        last_send = time.perf_counter() - t0
        await asyncio.sleep(1.0)
        await ws.close()
        await task

    after = [r for r in replies if r["_t"] > last_send]
    gaps = sorted(b["_t"] - a["_t"] for a, b in zip(replies, replies[1:]))
    preds = [r for r in replies if r["type"] == "prediction"]
    return {
        "hello": hello["type"],
        "replies": len(replies),
        "median_gap": gaps[len(gaps) // 2] if gaps else None,
        "drain": (after[-1]["_t"] - last_send) if after else 0.0,
        "queued_after": len(after),
        "accepted": [r["label"] for r in preds if r["accepted"]],
        "peak": max(preds, key=lambda r: r["confidence"]) if preds else None,
    }


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, json.loads(r.read())


# ---------------------------------------------------------------- checks

async def test_recognition(ws_url, clips):
    print(f"\nstreaming {len(clips)} held-out clips at {SEND_FPS} fps\n")
    print(f"  {'clip':<16}{'accepted':<26}{'peak':<18}{'gap':>6}{'drain':>7}")
    results = []
    for clip in clips:
        truth = clip.parent.name
        frames, fps = encode_clip(clip)
        r = await stream(ws_url, frames, fps)
        r["truth"] = truth
        results.append(r)
        peak = f"{r['peak']['label']} {r['peak']['confidence']:.2f}" if r["peak"] else "-"
        mark = "ok " if truth in r["accepted"] else "   "
        print(f"{mark}{truth + '/' + clip.stem:<16}{' '.join(r['accepted']) or '-':<26}{peak:<18}"
              f"{1000 * (r['median_gap'] or 0):>5.0f}ms{1000 * r['drain']:>5.0f}ms")

    print()
    n = len(results)
    hits = sum(r["truth"] in r["accepted"] for r in results)
    peak_hits = sum(r["peak"] and r["peak"]["label"] == r["truth"] for r in results)
    wrong = sum(sum(w != r["truth"] for w in r["accepted"]) for r in results)
    dupes = sum(len(set(r["accepted"])) < len(r["accepted"]) for r in results)
    peaks = {r["peak"]["label"] for r in results if r["peak"]}
    chance = 1 / 12

    check("every clip got a ready handshake and predictions",
          all(r["hello"] == "ready" and r["peak"] for r in results))
    check("latency stays bounded (median reply gap < 400 ms)",
          all(r["median_gap"] < 0.4 for r in results),
          f"worst {1000 * max(r['median_gap'] for r in results):.0f} ms")
    check("no backlog after the stream ends (<= 2 replies, < 1 s)",
          all(r["queued_after"] <= 2 and r["drain"] < 1.0 for r in results),
          f"worst {max(r['queued_after'] for r in results)} replies / "
          f"{1000 * max(r['drain'] for r in results):.0f} ms")
    check("model is not collapsed onto one class", len(peaks) >= 4, f"{len(peaks)} distinct peak labels")
    check("live accuracy well above chance (>= 2x)", peak_hits / n >= 2 * chance,
          f"top-1 at peak {peak_hits}/{n} = {peak_hits / n:.0%}, chance {chance:.0%}")
    check("no word emitted twice for one signing", dupes == 0, f"{dupes} clip(s) duplicated")
    print(f"\n  info: correct word emitted for {hits}/{n} clips ({hits / n:.0%}); "
          f"{wrong} wrong word(s) emitted in total")


async def test_reset(ws_url, clip):
    frames, _ = encode_clip(clip)
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.recv()
        for f in frames[:12]:
            await ws.send(json.dumps({"frame": f}))
            await asyncio.sleep(1 / SEND_FPS)
        await ws.send(json.dumps({"action": "reset"}))
        got_reset, after = False, None
        deadline = time.perf_counter() + 5
        while time.perf_counter() < deadline:
            m = json.loads(await asyncio.wait_for(ws.recv(), 5))
            if m["type"] == "reset":
                got_reset = True
                await ws.send(json.dumps({"frame": frames[12]}))
            elif got_reset:
                after = m
                break
    check("reset clears the window",
          got_reset and after and after["type"] == "buffering" and after["filled"] == 0,
          f"after reset: {after}")


async def test_malformed(ws_url, clip):
    frames, _ = encode_clip(clip)
    bad = [
        json.dumps({"frame": "not-base64!!"}),
        json.dumps({"frame": "data:image/jpeg;base64,"}),
        json.dumps({"frame": "data:image/jpeg;base64,AAAA"}),
        json.dumps({"something": "else"}),
        json.dumps({"frame": 42}),
        "42",
        "[1, 2]",
        "plain text that is not json",
    ]
    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.recv()
        for b in bad:
            await ws.send(b)
            await asyncio.sleep(0.3)  # let each be processed on its own
        await ws.send(json.dumps({"frame": frames[0]}))
        try:
            m = json.loads(await asyncio.wait_for(ws.recv(), 10))
            ok, detail = m["type"] in ("buffering", "prediction"), f"then got {m['type']}"
        except (websockets.ConnectionClosed, asyncio.TimeoutError) as e:
            ok, detail = False, f"connection lost: {type(e).__name__}"
    check("malformed frames are ignored, connection survives", ok, detail)


async def test_concurrent(ws_url, clips):
    a, b = (encode_clip(c) for c in clips[:2])
    ra, rb = await asyncio.gather(stream(ws_url, *a), stream(ws_url, *b))
    check("two concurrent clients both served with bounded lag",
          all(r["peak"] and r["median_gap"] < 0.8 and r["drain"] < 1.5 for r in (ra, rb)),
          f"gaps {1000 * ra['median_gap']:.0f}/{1000 * rb['median_gap']:.0f} ms, "
          f"drain {1000 * ra['drain']:.0f}/{1000 * rb['drain']:.0f} ms")


async def test_abrupt_disconnect(ws_url, http, clip):
    frames, _ = encode_clip(clip)
    for _ in range(3):
        ws = await websockets.connect(ws_url, max_size=None)
        await ws.recv()
        for f in frames[:5]:
            await ws.send(json.dumps({"frame": f}))
        ws.transport.abort()  # no close handshake, mid-processing
    await asyncio.sleep(1.5)
    status, _ = get(f"{http}/api/health")
    async with websockets.connect(ws_url, max_size=None) as ws:
        hello = json.loads(await ws.recv())["type"]
    check("server survives abrupt disconnects", status == 200 and hello == "ready")


def main():
    clips = held_out_clips()
    if not clips or not clips[0].exists():
        sys.exit("No held-out clips in data/raw -- run `python -m ml.fetch_wlasl` first.")

    port = free_port()
    log_path = BASE_DIR / "tests" / "live_test_server.log"
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"}
    with open(log_path, "w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
            cwd=BASE_DIR, stdout=log, stderr=subprocess.STDOUT, env=env,
        )
    http, ws_url = f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}/ws/recognize"
    try:
        for _ in range(120):
            try:
                get(f"{http}/api/health")
                break
            except OSError:
                time.sleep(0.5)
        status, health = get(f"{http}/api/health")
        print("live test\n")
        check("server starts with a trained model", status == 200 and health["model_trained"],
              f"{len(health['signs'])} signs")

        asyncio.run(test_recognition(ws_url, clips))
        print()
        asyncio.run(test_reset(ws_url, clips[0]))
        asyncio.run(test_malformed(ws_url, clips[0]))
        asyncio.run(test_concurrent(ws_url, clips))
        asyncio.run(test_abrupt_disconnect(ws_url, http, clips[0]))
    finally:
        server.terminate()
        server.wait(10)

    tracebacks = log_path.read_text(encoding="utf-8", errors="replace").count("Traceback")
    check("no tracebacks in the server log", tracebacks == 0, f"{tracebacks} found, see {log_path.name}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all live checks passed")


if __name__ == "__main__":
    main()
