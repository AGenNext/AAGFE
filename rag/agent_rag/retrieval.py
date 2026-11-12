"""Retrievers and query planner.

Each retriever returns ranked ``Hit`` objects; fusion uses Reciprocal Rank
Fusion (RRF) so scores from incomparable scales (cosine, BM25, graph weight)
can be combined without calibration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .contracts import RetrievalMode, RetrievalPlan
from .embeddings import cosine
from .store import IndexStore
from .text import tokenize

RRF_K = 60


@dataclass
class Hit:
    key: str  # chunk_id or community_id
    kind: Literal["chunk", "community_report"]
    score: float
    retrievers: list[str] = field(default_factory=list)


def _top(scores: dict[str, float], n: int, name: str, kind: str = "chunk") -> list[Hit]:
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [Hit(key=k, kind=kind, score=s, retrievers=[name]) for k, s in ranked]  # type: ignore[arg-type]


def rrf(*rankings: list[Hit], n: int) -> list[Hit]:
    fused: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking):
            cur = fused.setdefault(h.key, Hit(key=h.key, kind=h.kind, score=0.0))
            cur.score += 1.0 / (RRF_K + rank + 1)
            for r in h.retrievers:
                if r not in cur.retrievers:
                    cur.retrievers.append(r)
    return sorted(fused.values(), key=lambda h: (-h.score, h.key))[:n]


class Retrievers:
    def __init__(self, store: IndexStore) -> None:
        self.store = store

    def vector(self, query: str, n: int) -> list[Hit]:
        q = self.store.embedder.embed([query])[0]
        scores = {cid: cosine(q, self.store.vector(cid)) for cid in self.store.chunks()}
        return _top({k: v for k, v in scores.items() if v > 0}, n, "vector")

    def lexical(self, query: str, n: int) -> list[Hit]:
        return _top(self.store.bm25(query), n, "lexical")

    def hybrid(self, query: str, n: int) -> list[Hit]:
        return rrf(self.vector(query, n * 3), self.lexical(query, n * 3), n=n)

    # -- graph -------------------------------------------------------------

    def match_entities(self, query: str) -> list[str]:
        ql = query.lower()
        qtok = set(tokenize(query, keep_stopwords=True))
        out = []
        for eid, e in self.store.entities().items():
            name = e.name.lower()
            if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", ql):
                out.append(eid)
            elif "-" in name and set(name.split("-")) <= qtok and len(name.split("-")) > 1:
                out.append(eid)
        # Prefer the most specific names (Agent-Cognitive-Guard over Agent-Guard).
        return sorted(out, key=lambda e: (-len(e), e))

    def graph_local(self, query: str, n: int) -> tuple[list[Hit], list[str]]:
        seeds = self.match_entities(query)
        ents = self.store.entities()
        weights: dict[str, float] = {s: 1.0 for s in seeds}
        if seeds:
            for (a, b), rel in self.store.relations().items():
                for x, y in ((a, b), (b, a)):
                    if x in weights and weights[x] == 1.0 and y not in seeds:
                        weights[y] = max(weights.get(y, 0.0), min(0.5, 0.1 * rel.weight))
        chunk_scores: dict[str, float] = {}
        for eid, w in weights.items():
            for cid in ents[eid].chunk_ids:
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + w
        graph_rank = _top(chunk_scores, n * 3, "graph_local")
        # Graph neighbourhood gives recall around entities; hybrid keeps the
        # ranking anchored to the actual question.
        return rrf(graph_rank, self.hybrid(query, n * 3), n=n), seeds

    def graph_global(self, query: str, n: int) -> list[Hit]:
        comms = self.store.communities()
        if not comms:
            return []
        q = self.store.embedder.embed([query])[0]
        qt = set(tokenize(query))
        texts = {cid: f"{c.title}. {c.summary}" for cid, c in comms.items()}
        vecs = dict(zip(texts, self.store.embedder.embed(list(texts.values()))))
        max_rank = max(c.rank for c in comms.values()) or 1.0
        scores = {}
        for cid, c in comms.items():
            overlap = len(qt & set(tokenize(texts[cid]))) / (len(qt) or 1)
            scores[cid] = 0.5 * cosine(q, vecs[cid]) + 0.3 * overlap + 0.2 * (c.rank / max_rank)
        reports = _top(scores, max(1, min(3, n)), "graph_global", kind="community_report")
        # Map-reduce flavour: community reports for the big picture, plus the
        # best grounding chunks so answers still cite primary sources.
        return reports + self.hybrid(query, max(1, n - len(reports)))


GLOBAL_CUES = re.compile(
    r"\b(overall|overview|themes?|summari[sz]e|across|landscape|big picture|main (ideas|topics|components)|"
    r"what are (all|the) (components|modules|pillars)|holistic|high[- ]level)\b",
    re.I,
)


class QueryPlanner:
    """Chooses a retrieval strategy. Explicit modes always win over auto."""

    def __init__(self, retrievers: Retrievers) -> None:
        self.r = retrievers

    def plan(self, query: str, mode: RetrievalMode) -> RetrievalPlan:
        matched = self.r.match_entities(query)
        if mode is not RetrievalMode.auto:
            return RetrievalPlan(
                mode=mode, strategies=self._strategies(mode), rationale="explicit mode requested", matched_entities=matched
            )
        if GLOBAL_CUES.search(query):
            m, why = RetrievalMode.graph_global, "global/sensemaking cue detected in query"
        elif matched:
            m, why = RetrievalMode.graph_local, f"query names {len(matched)} known entit{'y' if len(matched) == 1 else 'ies'}"
        else:
            m, why = RetrievalMode.hybrid, "no entity or global cue; hybrid vector + lexical"
        return RetrievalPlan(mode=m, strategies=self._strategies(m), rationale=why, matched_entities=matched)

    @staticmethod
    def _strategies(mode: RetrievalMode) -> list[str]:
        return {
            RetrievalMode.vector: ["vector"],
            RetrievalMode.lexical: ["lexical"],
            RetrievalMode.hybrid: ["vector", "lexical", "rrf"],
            RetrievalMode.graph_local: ["entity-match", "1-hop-expansion", "vector", "lexical", "rrf"],
            RetrievalMode.graph_global: ["community-reports", "vector", "lexical", "rrf"],
        }.get(mode, [])

    def execute(self, query: str, plan: RetrievalPlan, n: int) -> list[Hit]:
        m = plan.mode
        if m is RetrievalMode.vector:
            return self.r.vector(query, n)
        if m is RetrievalMode.lexical:
            return self.r.lexical(query, n)
        if m is RetrievalMode.graph_local:
            return self.r.graph_local(query, n)[0]
        if m is RetrievalMode.graph_global:
            return self.r.graph_global(query, n)
        return self.r.hybrid(query, n)
