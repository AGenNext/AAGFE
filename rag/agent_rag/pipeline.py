"""Agent-Rag pipeline: plan -> retrieve -> provenance -> guard -> envelope.

    query planning
      -> local / global / hybrid retrieval
      -> trust-tier filter + provenance hashes      (Agent-Trust)
      -> Agent-Guard + Agent-Cognitive-Guard checks
      -> context candidate (trusted = False)        (-> Agent-Context)
      -> optional grounded generation               (via model-gateway)
    every stage -> Agent-Traces

Rule: retrieval output is not automatically trusted context.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from .contracts import (
    ContextCandidate,
    Evidence,
    GroundedAnswer,
    GuardDecision,
    RetrievalEnvelope,
    RetrievalRequest,
    SourceDocument,
    TrustTier,
)
from .generation import ExtractiveGenerator, Generator, citations_for, groundedness
from .guards import AgentGuard, CognitiveGuard, Guard, apply_guards
from .retrieval import Hit, QueryPlanner, Retrievers
from .store import IndexStore, InMemoryIndexStore
from .text import estimate_tokens, sha256
from .traces import HashChainTraceSink

_TIER_RANK = {TrustTier.external: 0, TrustTier.internal: 1, TrustTier.verified: 2}


class RagPipeline:
    def __init__(
        self,
        store: IndexStore | None = None,
        guards: list[Guard] | None = None,
        generator: Generator | None = None,
        traces: HashChainTraceSink | None = None,
    ) -> None:
        self.store = store or InMemoryIndexStore()
        self.retrievers = Retrievers(self.store)
        self.planner = QueryPlanner(self.retrievers)
        self.guards = guards if guards is not None else [AgentGuard(), CognitiveGuard(self.store)]
        self.generator = generator or ExtractiveGenerator()
        self.traces = traces or HashChainTraceSink()

    # -- index (stand-in for Agent-Knowledge sync) -------------------------

    def index(self, doc: SourceDocument, agent_id: str = "system:indexer") -> int:
        chunks = self.store.upsert_source(doc)
        self.traces.emit(
            f"idx-{uuid.uuid4().hex[:12]}", "index.upsert", agent_id,
            {"source_id": doc.source_id, "chunks": len(chunks), "trust_tier": doc.trust_tier.value,
             "content_hash": sha256(doc.text)},
        )
        return len(chunks)

    def index_directory(self, path: str | Path, trust: dict[str, TrustTier] | None = None) -> int:
        total = 0
        for p in sorted(Path(path).glob("*.md")):
            text = p.read_text(encoding="utf-8")
            title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), p.stem)
            tier = (trust or {}).get(p.stem, TrustTier.internal)
            total += self.index(SourceDocument(source_id=p.stem, title=title, text=text, uri=str(p.name), trust_tier=tier))
        return total

    # -- retrieval ---------------------------------------------------------

    def _to_evidence(self, hit: Hit) -> Evidence:
        if hit.kind == "community_report":
            c = self.store.communities()[hit.key]
            text = f"§ Community report: {c.title}\n{c.summary}"
            tiers = [self.store.chunks()[cid].trust_tier for cid in c.summary_chunk_ids if cid in self.store.chunks()]
            # A report is only as trustworthy as the weakest chunk it quotes.
            tier = min(tiers, key=_TIER_RANK.__getitem__, default=TrustTier.internal)
            return Evidence(evidence_id=hit.key, kind="community_report", text=text, title=f"Community report: {c.title}",
                            source_id=None, chunk_id=None, score=round(hit.score, 6), retrievers=hit.retrievers,
                            trust_tier=tier, content_hash=sha256(text))
        ch = self.store.chunks()[hit.key]
        return Evidence(evidence_id=ch.chunk_id, kind="chunk", text=ch.text, title=ch.title, source_id=ch.source_id,
                        chunk_id=ch.chunk_id, score=round(hit.score, 6), retrievers=hit.retrievers,
                        trust_tier=ch.trust_tier, content_hash=ch.content_hash)

    def retrieve(self, req: RetrievalRequest) -> RetrievalEnvelope:
        trace_id = f"rag-{uuid.uuid4().hex[:16]}"
        t = {}
        t0 = time.perf_counter()
        self.traces.emit(trace_id, "retrieve.request", req.agent_id,
                         {"query_hash": sha256(req.query), "mode": req.mode.value, "top_k": req.top_k, "team_id": req.team_id})

        plan = self.planner.plan(req.query, req.mode)
        t["plan"] = (time.perf_counter() - t0) * 1000
        self.traces.emit(trace_id, "retrieve.plan", req.agent_id, plan.model_dump(mode="json"))

        t1 = time.perf_counter()
        hits = self.planner.execute(req.query, plan, req.top_k)
        evidence = [self._to_evidence(h) for h in hits]
        if req.min_trust_tier:
            floor = _TIER_RANK[req.min_trust_tier]
            evidence = [e for e in evidence if _TIER_RANK[e.trust_tier] >= floor]
        t["retrieve"] = (time.perf_counter() - t1) * 1000
        self.traces.emit(trace_id, "retrieve.hits", req.agent_id,
                         {"evidence": [{"id": e.evidence_id, "hash": e.content_hash, "score": e.score} for e in evidence]})

        t2 = time.perf_counter()
        apply_guards(self.guards, req.query, evidence)
        t["guard"] = (time.perf_counter() - t2) * 1000
        allowed = [e for e in evidence if e.decision is GuardDecision.allow]
        quarantined = [e for e in evidence if e.decision is GuardDecision.quarantine]
        blocked = [e for e in evidence if e.decision is GuardDecision.block]
        self.traces.emit(trace_id, "retrieve.guard", req.agent_id, {
            "allowed": [e.evidence_id for e in allowed],
            "quarantined": [{"id": e.evidence_id, "rules": sorted({f.rule for f in e.findings})} for e in quarantined],
            "blocked": [{"id": e.evidence_id, "rules": sorted({f.rule for f in e.findings})} for e in blocked],
        })

        candidate = ContextCandidate(evidence_ids=[e.evidence_id for e in allowed],
                                     token_estimate=sum(estimate_tokens(e.text) for e in allowed))
        t["total"] = (time.perf_counter() - t0) * 1000
        env = RetrievalEnvelope(
            envelope_id=f"env-{uuid.uuid4().hex[:16]}", request=req, plan=plan, evidence=allowed,
            quarantined=quarantined, blocked=blocked, context_candidate=candidate, trace_id=trace_id,
            timings_ms={k: round(v, 2) for k, v in t.items()},
        )
        self.traces.emit(trace_id, "retrieve.handoff", req.agent_id,
                         {"envelope_id": env.envelope_id, "handoff_to": "Agent-Context", "trusted": False,
                          "evidence_ids": candidate.evidence_ids})
        return env

    def answer(self, req: RetrievalRequest) -> GroundedAnswer:
        env = self.retrieve(req)
        text = self.generator.generate(req.query, env.evidence)
        score = groundedness(text, len(env.evidence))
        self.traces.emit(env.trace_id, "generate.answer", req.agent_id,
                         {"generator": self.generator.name, "answer_hash": sha256(text), "groundedness": score})
        return GroundedAnswer(answer=text, citations=citations_for(env.evidence), generator=self.generator.name,
                              groundedness=score, envelope=env)
