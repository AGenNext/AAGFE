"""Embedding providers.

Agent-Rag does not own model routing, so embedders are a protocol. The
default ``HashingEmbedder`` is deterministic, dependency-free and offline: it
is good enough for demos and tests, and it is honest about being lexical
(feature hashing of unigrams + bigrams), not semantic. Swap in a
model-gateway-backed embedder for production semantic recall.
"""

from __future__ import annotations

import math
import zlib
from typing import Protocol, Sequence

from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # Inputs are L2-normalised by the embedder.
    return sum(x * y for x, y in zip(a, b))
