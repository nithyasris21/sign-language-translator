"""Train the sign classifier on recorded landmark sequences.

    python -m ml.train

Reads data/sequences/<label>/*.npy, writes data/models/sign_lstm.keras and
labels.json. Architecture is a small stacked LSTM -- with a few hundred
samples per class anything deeper just overfits.
"""
import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    FEATURE_SIZE,
    LABELS_PATH,
    MODEL_PATH,
    SEQUENCE_LENGTH,
    SEQ_DIR,
)


def load_dataset():
    labels = sorted(d.name for d in SEQ_DIR.iterdir() if d.is_dir() and any(d.glob("*.npy")))
    if len(labels) < 2:
        raise SystemExit("Need at least 2 signs. Run `python -m ml.collect <word>` "
                         "or `python -m ml.fetch_wlasl` + `python -m ml.preprocess`.")

    X, y, groups = [], [], []
    for idx, label in enumerate(labels):
        for path in sorted((SEQ_DIR / label).glob("*.npy")):
            seq = np.load(path)
            if seq.shape != (SEQUENCE_LENGTH, FEATURE_SIZE):
                print(f"skipping {path} with shape {seq.shape}")
                continue
            X.append(seq)
            y.append(idx)
            # "00423_m2.npy" -> "hello/00423": every augmented copy of one
            # source clip shares a group so the split cannot straddle it
            groups.append(f"{label}/{path.stem.rsplit('_', 1)[0]}")
    print(f"{len(X)} samples across {len(labels)} signs: {', '.join(labels)}")
    return np.array(X, dtype=np.float32), np.array(y), labels, np.array(groups)


def build_model(n_classes: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(SEQUENCE_LENGTH, FEATURE_SIZE)),
        layers.Masking(mask_value=0.0),
        layers.LSTM(64, return_sequences=True, activation="tanh"),
        layers.Dropout(0.3),
        layers.LSTM(128, return_sequences=True, activation="tanh"),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False, activation="tanh"),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30, help="early-stopping patience in epochs")
    args = ap.parse_args()

    X, y, labels, groups = load_dataset()

    # Group-aware split. Mirrored and time-shifted copies of the same clip are
    # near-duplicates: letting them span the split would report an accuracy
    # that collapses on any genuinely unseen signer.
    # Signs with a single source clip (text -> sign reference videos) can't be
    # held out without never being trained on, so they always go to training.
    clips_per_label = {}
    for label, group in zip(y, groups):
        clips_per_label.setdefault(label, set()).add(group)
    multi = np.array([len(clips_per_label[label]) > 1 for label in y])
    splitter = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=42)
    multi_idx = np.flatnonzero(multi)
    tr, test_idx = next(splitter.split(X[multi_idx], y[multi_idx], groups=groups[multi_idx]))
    train_idx = np.concatenate([multi_idx[tr], np.flatnonzero(~multi)])
    test_idx = multi_idx[test_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"{len(X_train)} train / {len(X_test)} held-out "
          f"({len(set(groups[test_idx]))} unseen source clips)")

    model = build_model(len(labels))
    model.summary()
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=max(3, args.patience // 3), factor=0.5, min_lr=1e-5),
        ],
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    probs = model.predict(X_test, verbose=0)
    top3 = float(np.mean([t in np.argsort(pr)[-3:] for t, pr in zip(y_test, probs)]))
    print(f"\nheld-out accuracy: {acc:.3f}  top-3: {top3:.3f}  loss: {loss:.3f}")
    print(f"chance baseline:   {1 / len(labels):.3f}")

    model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(labels, indent=2))
    print(f"saved {MODEL_PATH}\nsaved {LABELS_PATH}")


if __name__ == "__main__":
    main()
