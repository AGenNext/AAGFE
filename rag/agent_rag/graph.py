"""GraphRAG-inspired graph construction (entities, relations, communities).

Implements the *pattern* (entity graph -> communities -> community reports
for global questions, entity neighbourhoods for local questions). It does not
reuse Microsoft GraphRAG code. Extraction is pluggable: the default
``HeuristicEntityExtractor`` is deterministic and offline; an LLM extractor
via model-gateway can replace it without touching retrieval.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable, Protocol

from .contracts import Chunk, Community, Entity, Relation, TrustTier
from .text import is_heading, sentences, tokenize

_CAND = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)+"  # Agent-Rag, Agent-Cognitive-Guard
    r"|[A-Z]{2,}[A-Za-z0-9]*"  # OPA, AAGFE, RAG
    r"|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*"  # SurrealDB, OpenFGA, GraphRAG
    r"|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"  # Agent Vault, Open Policy Agent
)
_NOISE = frozenset(
    "The This That These Those It Its Agent Use When What Why How Each Every Must Should "
    "Add Replace Open Then Source Scope Decision Boundary Rule Component Responsibility "
    "Initial Latest Authors Venue Dimension Proposed Suggested Product Core Local JSON TODO".split()
)


def _norm(name: str) -> str:
    return re.sub(r"[\s_]+", "-", name.strip()).lower()


class EntityExtractor(Protocol):
    def extract(self, text: str) -> list[str]: ...


class HeuristicEntityExtractor:
    """Proper-noun / product-name extractor.

    Strong forms (hyphenated names, acronyms, CamelCase) are always kept;
    multi-word Title Case spans are kept only when they recur, which filters
    most headings. See ``build_graph(min_mentions=...)``.
    """

    def extract(self, text: str) -> list[str]:
        out: list[str] = []
        for m in _CAND.finditer(text):
            name = m.group(0).strip()
            first = name.split()[0].split("-")[0]
            if name in _NOISE or (" " in name and first in _NOISE and len(name.split()) == 2 and name.split()[1] in _NOISE):
                continue
            out.append(name)
        return out


def _is_strong(name: str) -> bool:
    return "-" in name or name.isupper() or bool(re.search(r"[a-z][A-Z]", name))


def build_graph(
    chunks: Iterable[Chunk],
    extractor: EntityExtractor | None = None,
    *,
    min_mentions: int = 2,
) -> tuple[dict[str, Entity], dict[tuple[str, str], Relation]]:
    extractor = extractor or HeuristicEntityExtractor()
    display: dict[str, Counter[str]] = defaultdict(Counter)
    mentions: Counter[str] = Counter()
    in_chunks: dict[str, set[str]] = defaultdict(set)
    per_chunk: dict[str, set[str]] = {}

    chunks = list(chunks)
    for ch in chunks:
        names = extractor.extract(ch.text)
        ids = set()
        for n in names:
            eid = _norm(n)
            display[eid][n] += 1
            mentions[eid] += 1
            in_chunks[eid].add(ch.chunk_id)
            ids.add(eid)
        per_chunk[ch.chunk_id] = ids

    keep = {
        eid
        for eid, c in mentions.items()
        if c >= min_mentions or _is_strong(display[eid].most_common(1)[0][0])
    }
    entities = {
        eid: Entity(
            entity_id=eid,
            name=display[eid].most_common(1)[0][0],
            mentions=mentions[eid],
            chunk_ids=sorted(in_chunks[eid]),
        )
        for eid in sorted(keep)
    }

    relations: dict[tuple[str, str], Relation] = {}
    for cid, ids in per_chunk.items():
        kept = sorted(ids & keep)
        for i, a in enumerate(kept):
            for b in kept[i + 1 :]:
                rel = relations.setdefault((a, b), Relation(source=a, target=b, weight=0.0))
                rel.weight += 1.0
                rel.chunk_ids.append(cid)
    return entities, relations


def detect_communities(
    entities: dict[str, Entity], relations: dict[tuple[str, str], Relation], *, iterations: int = 25
) -> dict[str, str]:
    """Deterministic weighted label propagation. Returns entity_id -> label."""
    adj: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), r in relations.items():
        # Normalise by endpoint frequency so hub entities (the product name
        # mentioned everywhere) don't pull every node into one community.
        w = r.weight / ((entities[a].mentions * entities[b].mentions) ** 0.5)
        adj[a][b] = adj[a].get(b, 0.0) + w
        adj[b][a] = adj[b].get(a, 0.0) + w
    label = {e: e for e in entities}
    order = sorted(entities, key=lambda e: (-entities[e].mentions, e))
    for _ in range(iterations):
        changed = False
        for node in order:
            if not adj[node]:
                continue
            score: dict[str, float] = defaultdict(float)
            for nb, w in adj[node].items():
                score[label[nb]] += w
            best = max(sorted(score), key=lambda lab: score[lab])
            if score[best] > score.get(label[node], 0.0) and best != label[node]:
                label[node] = best
                changed = True
        if not changed:
            break
    return label


def build_communities(
    entities: dict[str, Entity],
    relations: dict[tuple[str, str], Relation],
    chunks_by_id: dict[str, Chunk],
    *,
    max_summary_sentences: int = 4,
) -> dict[str, Community]:
    """Group entities and write an extractive community report for each group.

    Reports are extractive (sentences lifted verbatim from member chunks), so
    every claim in a report remains traceable to a source hash.
    """
    labels = detect_communities(entities, relations)
    groups: dict[str, list[str]] = defaultdict(list)
    for eid, lab in labels.items():
        groups[lab].append(eid)

    communities: dict[str, Community] = {}
    for idx, (lab, members) in enumerate(
        sorted(groups.items(), key=lambda kv: -sum(entities[e].mentions for e in kv[1]))
    ):
        if len(members) < 2:
            continue
        members.sort(key=lambda e: (-entities[e].mentions, e))
        cid = f"community:{idx}"
        for e in members:
            entities[e].community_id = cid
        chunk_ids = sorted({c for e in members for c in entities[e].chunk_ids})
        names_lc = [entities[e].name.lower() for e in members]

        scored: list[tuple[int, int, str, str]] = []
        seen: set[str] = set()
        for pos, ch_id in enumerate(chunk_ids):
            # External-tier content may be retrieved (and guarded) as a chunk,
            # but it is never laundered into a community report.
            if chunks_by_id[ch_id].trust_tier is TrustTier.external:
                continue
            for s in sentences(chunks_by_id[ch_id].text):
                key = " ".join(tokenize(s))
                if key in seen or len(s) < 25 or is_heading(s):
                    continue
                seen.add(key)
                hits = sum(1 for n in names_lc if n in s.lower())
                if hits:
                    scored.append((hits, -pos, s, ch_id))
        scored.sort(reverse=True)
        picked = scored[:max_summary_sentences]
        summary = " ".join(s if s.endswith((".", "!", "?")) else s + "." for _, _, s, _ in picked)
        communities[cid] = Community(
            community_id=cid,
            entity_ids=members,
            title=", ".join(entities[e].name for e in members[:3]),
            summary=summary or "No representative sentences found.",
            chunk_ids=chunk_ids,
            summary_chunk_ids=sorted({c for *_, c in picked}),
            rank=float(sum(entities[e].mentions for e in members)),
        )
    return communities
