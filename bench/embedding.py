"""A tiny deterministic embedding so the bench needs no model download.

Words are hashed into a fixed number of buckets, and the vector is normalized. Texts that share
words end up close together, which is enough to show how similarity search behaves.
"""

from __future__ import annotations

import hashlib
import math
import re

DIMS = 64
STOP_WORDS = {"what", "is", "the", "under", "in", "of", "a", "an", "and", "for", "to", "see"}


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP_WORDS]


def embed(text: str) -> list[float]:
    vec = [0.0] * DIMS
    for word in tokens(text):
        digest = hashlib.md5(word.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % DIMS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[index] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
