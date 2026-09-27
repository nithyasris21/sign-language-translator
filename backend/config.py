import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SEQ_DIR = DATA_DIR / "sequences"      # raw recorded landmark sequences
MODEL_DIR = DATA_DIR / "models"       # trained keras model + label list
SIGN_DIR = DATA_DIR / "signs"         # reference clips for text -> sign

MODEL_PATH = MODEL_DIR / "sign_lstm.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

# Pretrained 250-sign model: 1st place of Google's Kaggle "ASL Signs" (ISLR)
# competition, from huggingface.co/sign/kaggle-asl-signs-1st-place (MIT).
# Trained on ~94k takes by 21 signers, so it generalises to new signers far
# better than anything WLASL's ~10 clips per word can give.
ISLR_DIR = MODEL_DIR / "islr"
ISLR_MODEL_PATH = ISLR_DIR / "model.tflite"
ISLR_LABELS_PATH = ISLR_DIR / "sign_to_prediction_index_map.json"
ISLR_FPS = 30              # its training videos were ~30 fps; live frames are resampled to match
ISLR_WINDOW_SECONDS = 1.5  # median ISLR take is ~1.3 s

# "islr" (pretrained, default when downloaded) or "lstm" (trained here by ml.train)
SIGN_MODEL = os.environ.get("SIGN_MODEL", "islr" if ISLR_MODEL_PATH.exists() else "lstm")

SEQUENCE_LENGTH = 30      # frames per sample
WINDOW_SECONDS = 2.0      # live time span resampled to SEQUENCE_LENGTH (median WLASL clip ~2.1 s)
FEATURE_SIZE = 258        # pose(132) + left hand(63) + right hand(63)
SEQUENCES_PER_SIGN = 40   # how many samples to record per word
CONFIDENCE_THRESHOLD = 0.75

for d in (SEQ_DIR, MODEL_DIR, SIGN_DIR):
    d.mkdir(parents=True, exist_ok=True)
