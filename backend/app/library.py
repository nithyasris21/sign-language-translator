"""Text -> sign lookup.

data/signs/
    words/<word>.mp4|webm|gif     one clip per vocabulary word
    letters/<a-z>.png|jpg|gif     fingerspelling fallback
"""
import re

from config import SIGN_DIR

WORDS_DIR = SIGN_DIR / "words"
LETTERS_DIR = SIGN_DIR / "letters"
CLIP_EXT = (".mp4", ".webm", ".gif", ".png", ".jpg", ".jpeg")

for d in (WORDS_DIR, LETTERS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _index(directory):
    return {p.stem.lower(): p for p in directory.iterdir() if p.suffix.lower() in CLIP_EXT}


def available_words() -> list[str]:
    return sorted(_index(WORDS_DIR))


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[^A-Za-z0-9']+", text.lower()) if t]


def translate(text: str) -> list[dict]:
    """Turn a sentence into an ordered list of renderable sign items.

    A word with its own clip is signed directly; anything else is
    fingerspelled letter by letter, which is what a human signer does too.
    """
    words = _index(WORDS_DIR)
    letters = _index(LETTERS_DIR)
    out: list[dict] = []

    for token in tokenize(text):
        if token in words:
            out.append({
                "type": "word",
                "label": token,
                "src": f"/media/words/{words[token].name}",
            })
            continue

        spelled = []
        for ch in token:
            if ch in letters:
                spelled.append({
                    "type": "letter",
                    "label": ch.upper(),
                    "src": f"/media/letters/{letters[ch].name}",
                })
        if spelled:
            out.append({"type": "fingerspell", "label": token, "items": spelled})
        else:
            out.append({"type": "missing", "label": token, "src": None})

    return out
