# Sign Language Translator

Two-way translation between hand signs and text.

- **Sign → Text**: webcam frames stream to a FastAPI WebSocket, MediaPipe Holistic
  extracts pose + hand landmarks, and an LSTM classifies 30-frame windows.
- **Text → Sign**: typed text is tokenised, known words play their reference clip,
  and unknown words are fingerspelled letter by letter.

```
signlang/
  backend/
    app/main.py        FastAPI: /api/health, /api/text-to-sign, /ws/recognize
    app/library.py     text -> sign lookup + fingerspelling fallback
    ml/landmarks.py    MediaPipe Holistic -> 258-dim feature vector
    ml/collect.py      record training sequences from the webcam
    ml/train.py        train + save the LSTM
    ml/predict.py      sliding-window inference with debouncing
    ml/record_clip.py  record a reference clip for text -> sign
    data/              sequences/ models/ signs/   (generated)
  frontend/            Vite + React
```

## Why these choices

**Landmarks, not raw pixels.** Feeding video straight into a CNN needs tens of
thousands of clips. Landmarks throw away lighting, background and clothing, so a
few dozen takes per sign is enough — and a 258-float vector trains on a laptop CPU.

**Sequences, not single frames.** Many signs differ only in motion (in ASL,
"please" and "sorry" are near-identical handshapes with different movement), so
the model sees a 30-frame window, roughly 1.5 s.

**Face mesh excluded.** Its 468 points would swamp the 126 hand values. Add it
back in `extract_keypoints` if you need signs that depend on facial grammar.

## Setup

MediaPipe and TensorFlow do **not** support Python 3.13 — use 3.12. The venv is
already created at `backend/.venv`; to rebuild it from scratch:

```bash
cd backend
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --timeout 180 --retries 10 -r requirements.txt
```

Do not run `pip install --upgrade pip` inside this venv on Windows — pip
replacing itself mid-run leaves a broken `~ip` directory and the venv has to be
recreated.

```bash
cd frontend
npm install
```

## Verify the install

Before spending an hour recording takes, confirm the stack works:

```bash
cd backend
.venv\Scripts\Activate.ps1
python -m tests.smoke_test
```

It checks MediaPipe produces a 258-dim vector, the LSTM can overfit separable
synthetic sequences, the sliding window and debounce behave, and both HTTP
routes respond — no webcam or recorded data required.

A model trained on 12 WLASL glosses is included in `backend/data/models/`, so
Sign → Text works straight after install. Text → Sign needs reference clips in
`data/signs/`, which `ml.fetch_wlasl` downloads (they are not in the repo).

## Tests

| Command | Needs | Covers |
|---|---|---|
| `cd backend && python -m tests.smoke_test` | install only | MediaPipe, LSTM, windowing/debounce, HTTP routes |
| `cd backend && python -m tests.live_test` | model + `data/raw` | real server; held-out clips streamed at 12 fps: latency, accuracy, reset, malformed frames, concurrent clients, disconnects |
| `cd frontend && npm run test:ui` | model + `data/raw` + Chrome | the React app in Chrome with a clip as fake webcam: both tabs, all controls, error states, production build |

## Pretrained model (default)

Sign → Text uses the 1st-place model of Google's Kaggle
[ASL Signs](https://www.kaggle.com/competitions/asl-signs) competition,
from [sign/kaggle-asl-signs-1st-place](https://huggingface.co/sign/kaggle-asl-signs-1st-place)
(MIT). It recognises 250 everyday signs and was trained on ~94k takes by 21
signers, so it works on new signers without any recording: 76% top-1 / 91%
top-5 on WLASL clips of its vocabulary, fed at the live ~5 fps rate.

```bash
mkdir data\models\islr
curl -L -o data/models/islr/model.tflite https://huggingface.co/sign/kaggle-asl-signs-1st-place/resolve/main/model.tflite
curl -L -o data/models/islr/sign_to_prediction_index_map.json https://huggingface.co/sign/kaggle-asl-signs-1st-place/resolve/main/sign_to_prediction_index_map.json
```

It is used whenever those files exist. Set `SIGN_MODEL=lstm` to use the
model trained by `ml.train` instead (e.g. for words outside the 250, such as
numbers).

## Build your vocabulary

## Option A — train on WLASL (no webcam needed)

[WLASL](https://huggingface.co/datasets/Voxel51/WLASL) is 11,880 labelled
word-level ASL videos (CC BY-NC 4.0, research use).

```bash
python -m ml.fetch_wlasl --top 12     # downloads clips + text->sign reference clips
python -m ml.preprocess               # videos -> landmark sequences (parallel)
python -m ml.train
```

`--top 12` takes the glosses with the **most clips**, not the most useful
words. WLASL has 2000 glosses but a median of 6 clips each, so sample count
decides which classes are learnable at all — which is why "hello" and "thanks"
are not in the default set. Pick your own with `--words drink help go`.

Two things the preprocessor does that matter:

- **Augmentation.** ~14 clips per gloss is far too few, so each clip yields 8
  sequences: mirrored (a left-handed signing) × 4 temporal crops (tolerance to
  signing speed). Mirroring re-runs MediaPipe on the flipped frame instead of
  negating x in the feature vector, because pose landmarks are indexed by body
  side and a naive flip would swap their meaning.
- **Parallelism.** Holistic manages ~1.5 fps per core; single-threaded this is
  hours. Each worker process gets its own Holistic instance — the object is
  neither picklable nor thread-safe.

Training splits with `StratifiedGroupKFold` grouped by source clip. The
augmented copies are near-duplicates, so a plain random split would put copies
of one clip on both sides and report an accuracy that collapses on any real
unseen signer.

### What accuracy to expect

Low, and that is the dataset, not the pipeline. ~14 clips per class spread over
many different signers is genuinely hard — published WLASL100 top-1 sits around
60–70% with far heavier models than this LSTM. Read the reported number against
the printed chance baseline, and treat webcam data (Option B) as the path to a
system that actually works for one signer.

## Option B — record your own signs


1. **Record** 40 takes each for at least two signs:
   ```bash
   python -m ml.collect hello
   python -m ml.collect thanks
   python -m ml.collect yes
   ```
   Vary distance, angle and speed between takes — the model generalises only as
   far as your recordings do. Press `q` to abort.

2. **Train**:
   ```bash
   python -m ml.train
   ```
   Prints held-out accuracy and writes `data/models/sign_lstm.keras`.

3. **Reference clips** for the reverse direction:
   ```bash
   python -m ml.record_clip hello
   ```
   For fingerspelling, drop `a.png` … `z.png` into `data/signs/letters/`.

## Run

```bash
cd backend && uvicorn app.main:app --reload --port 8000
cd frontend && npm run dev            # http://localhost:5173
```

## Tuning

All in `backend/config.py`:

| Setting | Effect |
|---|---|
| `SEQUENCE_LENGTH` | Frames per model input (must match what the model was trained on) |
| `WINDOW_SECONDS` | Live time span resampled to `SEQUENCE_LENGTH`; match your training clips' length |
| `SEQUENCES_PER_SIGN` | More takes per sign = better generalisation |
| `CONFIDENCE_THRESHOLD` | Raise if words fire spuriously, lower if signs are missed |

The server runs Holistic at whatever rate the CPU allows (~4–6 fps on a
laptop), keeps only the newest frame when the client sends faster, and
resamples the last `WINDOW_SECONDS` by timestamp, so latency stays bounded and
the model sees the same time span regardless of machine speed. `SEND_FPS` in
`frontend/src/components/SignToText.jsx` only needs to exceed that rate.

If you change `ml/landmarks.py`, re-run `python -m ml.preprocess --fresh` and
`python -m ml.train`: a model trained on the old feature layout will load and
run without error but predict one class for everything.

## Known limits

- Accuracy drops hard on signers, lighting or camera angles unlike your training
  takes. Record from multiple people if this needs to work for anyone but you.
- No continuous-signing segmentation: the debounce in `FrameBuffer.accept`
  emits a word when confidence crosses the threshold, which is a heuristic, not
  a proper CTC-style boundary detector.
- Text → sign is a lookup, not generation. It plays back clips you recorded and
  ignores grammar (ASL word order differs from English).