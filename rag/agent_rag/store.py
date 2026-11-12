"""Index store.

Agent-Rag reads from indexes; it does not own long-term knowledge storage.
``IndexStore`` is the seam where a SurrealDB adapter (graph + vector indexes,
as named in the Agent-Rag charter) plugs in. ``InMemoryIndexStore`` is the
reference implementation used by the service and the tests.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Protocol

from .contracts import Chunk, Community, Entity, Relation, SourceDocument
from .embeddings import Embedder, HashingEmbedder
from .graph import EntityExtractor, build_communities, build_graph
from .text import chunk_text, sha256, tokenize


class IndexStore(Protocol):
    embedder: Embedder

    def upsert_source(self, doc: SourceDocument) -> list[Chunk]: ...
    def chunks(self) -> dict[str, Chunk]: ...
    def vector(self, chunk_id: str) -> list[float]: ...
    def bm25(self, query: str) -> dict[str, float]: ...
    def entities(self) -> dict[str, Entity]: ...
    def relations(self) -> dict[tuple[str, str], Relation]: ...
    def communities(self) -> dict[str, Community]: ...
    def sources(self) -> dict[str, SourceDocument]: ...


class InMemoryIndexStore:
    def __init__(
        self,
        embedder: Embedder | None = None,
        extractor: EntityExtractor | None = None,
        *,
        chunk_words: int = 120,
        overlap_words: int = 20,
    ) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._extractor = extractor
        self._chunk_words = chunk_words
        self._overlap = overlap_words
        self._lock = threading.RLock()
        self._sources: dict[str, SourceDocument] = {}
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._df: Counter[str] = Counter()
        self._entities: dict[str, Entity] = {}
        self._relations: dict[tuple[str, str], Relation] = {}
        self._communities: dict[str, Community] = {}

    # -- writes ------------------------------------------------------------

    def upsert_source(self, doc: SourceDocument) -> list[Chunk]:
        with self._lock:
            self._remove_source(doc.source_id)
            self._sources[doc.source_id] = doc
            texts = chunk_text(doc.text, max_words=self._chunk_words, overlap_words=self._overlap)
            new = [
                Chunk(
                    chunk_id=f"{doc.source_id}#{i}",
                    source_id=doc.source_id,
                    title=doc.title,
                    text=t,
                    ordinal=i,
                    trust_tier=doc.trust_tier,
                    content_hash=sha256(t),
                )
                for i, t in enumerate(texts)
            ]
            vecs = self.embedder.embed([c.text for c in new])
            for c, v in zip(new, vecs):
                self._chunks[c.chunk_id] = c
                self._vectors[c.chunk_id] = v
                tf = Counter(tokenize(c.text))
                self._tf[c.chunk_id] = tf
                self._df.update(tf.keys())
            self._rebuild_graph()
            return new

    def _remove_source(self, source_id: str) -> None:
        for cid in [c for c, ch in self._chunks.items() if ch.source_id == source_id]:
            self._df.subtract(self._tf.pop(cid).keys())
            del self._chunks[cid]
            del self._vectors[cid]
        self._df += Counter()  # drop zero / negative counts
        self._sources.pop(source_id, None)

    def delete_source(self, source_id: str) -> bool:
        with self._lock:
            existed = source_id in self._sources
            self._remove_source(source_id)
            self._rebuild_graph()
            return existed

    def _rebuild_graph(self) -> None:
        ents, rels = build_graph(self._chunks.values(), self._extractor)
        self._communities = build_communities(ents, rels, self._chunks)
        self._entities, self._relations = ents, rels

    # -- reads -------------------------------------------------------------

    def chunks(self) -> dict[str, Chunk]:
        return self._chunks

    def vector(self, chunk_id: str) -> list[float]:
        return self._vectors[chunk_id]

    def bm25(self, query: str, *, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        q = tokenize(query)
        if not q or not self._tf:
            return {}
        n = len(self._tf)
        avgdl = sum(sum(tf.values()) for tf in self._tf.values()) / n
        scores: dict[str, float] = {}
        for cid, tf in self._tf.items():
            dl = sum(tf.values())
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
            if s > 0:
                scores[cid] = s
        return scores

    def entities(self) -> dict[str, Entity]:
        return self._entities

    def relations(self) -> dict[tuple[str, str], Relation]:
        return self._relations

    def communities(self) -> dict[str, Community]:
        return self._communities

    def sources(self) -> dict[str, SourceDocument]:
        return self._sources
