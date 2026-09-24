from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SEQ_DIR = DATA_DIR / "sequences"      # raw recorded landmark sequences
MODEL_DIR = DATA_DIR / "models"       # trained keras model + label list
SIGN_DIR = DATA_DIR / "signs"         # reference clips for text -> sign

MODEL_PATH = MODEL_DIR / "sign_lstm.keras"
LABELS_PATH = MODEL_DIR / "labels.json"

SEQUENCE_LENGTH = 30      # frames per sample
WINDOW_SECONDS = 2.0      # live time span resampled to SEQUENCE_LENGTH (median WLASL clip ~2.1 s)
FEATURE_SIZE = 258        # pose(132) + left hand(63) + right hand(63)
SEQUENCES_PER_SIGN = 40   # how many samples to record per word
CONFIDENCE_THRESHOLD = 0.75

for d in (SEQ_DIR, MODEL_DIR, SIGN_DIR):
    d.mkdir(parents=True, exist_ok=True)
