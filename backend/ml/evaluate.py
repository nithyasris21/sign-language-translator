"""Score the saved model on the same held-out split train.py used.

    python -m ml.evaluate

Reproduces the split deterministically (same seed and grouping), so this
reports on source clips the model never saw -- no retraining required.
"""
import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow import keras

from config import LABELS_PATH, MODEL_PATH
from ml.train import load_dataset


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("No trained model. Run `python -m ml.train` first.")

    X, y, labels, groups = load_dataset()
    splitter = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
    _, test_idx = next(splitter.split(X, y, groups=groups))
    X_test, y_test = X[test_idx], y[test_idx]

    model = keras.models.load_model(MODEL_PATH)
    probs = model.predict(X_test, verbose=0)
    pred = probs.argmax(axis=1)

    top1 = float((pred == y_test).mean())
    top3 = float(np.mean([t in np.argsort(p)[-3:] for t, p in zip(y_test, probs)]))
    chance = 1 / len(labels)

    print(f"\nheld-out: {len(X_test)} sequences from {len(set(groups[test_idx]))} unseen clips")
    print(f"top-1 {top1:.3f}   top-3 {top3:.3f}   chance {chance:.3f} "
          f"({top1 / chance:.1f}x chance)\n")

    print(f"{'sign':<14}{'recall':>8}{'n':>6}")
    for i, label in enumerate(labels):
        mask = y_test == i
        if mask.sum():
            print(f"{label:<14}{(pred[mask] == i).mean():>8.2f}{mask.sum():>6}")

    print("\nmost common confusions:")
    pairs = {}
    for t, p in zip(y_test, pred):
        if t != p:
            pairs[(labels[t], labels[p])] = pairs.get((labels[t], labels[p]), 0) + 1
    for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {t} -> {p}: {n}")


if __name__ == "__main__":
    main()
