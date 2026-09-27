import asyncio
import json
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import library
from config import CONFIDENCE_THRESHOLD, SIGN_DIR
from ml.landmarks import (
    decode_data_url,
    detect,
    extract_keypoints,
    extract_raw543,
    has_hands,
    make_holistic,
)
from ml.predict import FrameBuffer, recognizer

app = FastAPI(title="Sign Language Translator")

# Vite dev server (5173) and `vite preview` of the production build (4173);
# set CORS_ORIGINS (comma-separated) when serving the frontend from elsewhere
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=SIGN_DIR), name="media")


class TextRequest(BaseModel):
    text: str


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_trained": recognizer.ready,
        "signs": recognizer.labels,
        "model": recognizer.kind,
        "sequence_length": recognizer.sequence_length,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


@app.get("/api/vocabulary")
def vocabulary():
    """What the system can recognise, and what it can display."""
    return {
        "recognizable": recognizer.labels,
        "displayable": library.available_words(),
    }


@app.post("/api/text-to-sign")
def text_to_sign(req: TextRequest):
    return {"text": req.text, "sequence": library.translate(req.text)}


@app.websocket("/ws/recognize")
async def recognize(ws: WebSocket):
    """Client streams base64 JPEG frames; we stream back predictions.

    MediaPipe Holistic is stateful per video stream, so each connection gets
    its own instance and its own rolling window.
    """
    await ws.accept()
    buffer = FrameBuffer(recognizer.window_seconds, recognizer.sequence_length)
    holistic = make_holistic()
    extract = extract_raw543 if recognizer.kind == "islr" else extract_keypoints

    def process(data_url: str, t: float) -> dict | None:
        """CPU-bound: runs in a worker thread so the event loop stays free."""
        frame = decode_data_url(data_url)
        if frame is None:
            return None
        results = detect(frame, holistic)
        window = buffer.push(extract(results), t)
        hands = has_hands(results)
        if window is None:
            return {"type": "buffering", "filled": buffer.filled, "needed": buffer.sequence_length, "hands": hands}
        label, confidence, scores = recognizer.predict(window)
        return {
            "type": "prediction",
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "hands": hands,
            "accepted": buffer.accept(label, confidence),
        }

    # Holistic runs slower than the client sends, so frames are not queued:
    # the receiver keeps only the newest one and the processor picks it up
    # when free. Queuing every frame makes latency grow without bound.
    state = {"frame": None, "reset": False, "closed": False}
    wake = asyncio.Event()

    async def receive():
        try:
            while True:
                message = await ws.receive_text()
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    payload = None
                if not isinstance(payload, dict):
                    payload = {"frame": message}
                if payload.get("action") == "reset":
                    state["reset"], state["frame"] = True, None
                elif isinstance(payload.get("frame"), str) and payload["frame"]:
                    state["frame"] = (payload["frame"], time.monotonic())
                wake.set()
        except WebSocketDisconnect:
            pass
        finally:
            state["closed"] = True
            wake.set()

    receiver = None
    try:
        if not recognizer.ready:
            await ws.send_json({"type": "error", "message": "No trained model yet. Run ml/train.py."})
            await ws.close()
            return

        await asyncio.to_thread(recognizer.load)
        await ws.send_json({"type": "ready", "signs": recognizer.labels})

        receiver = asyncio.create_task(receive())
        while True:
            await wake.wait()
            wake.clear()
            if state["closed"]:
                break
            if state["reset"]:
                state["reset"] = False
                buffer.reset()
                await ws.send_json({"type": "reset"})
            item, state["frame"] = state["frame"], None
            if item is None:
                continue
            msg = await asyncio.to_thread(process, *item)
            # the client may have hung up while Holistic was running;
            # sending after close raises in Starlette
            if state["closed"]:
                break
            if msg is not None:
                await ws.send_json(msg)

    except WebSocketDisconnect:
        pass
    finally:
        if receiver is not None:
            receiver.cancel()
        holistic.close()
