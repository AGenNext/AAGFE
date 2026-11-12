"""HTTP API for Agent-Rag.

Run:  uvicorn agent_rag.api:app --port 8088   (from the rag/ directory)
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .contracts import GroundedAnswer, RetrievalEnvelope, RetrievalRequest, SourceDocument, TraceEvent, TrustTier
from .generation import generator_from_env
from .pipeline import RagPipeline

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(os.environ.get("AGENT_RAG_CORPUS", ROOT / "corpus"))
# Seed-corpus trust tiers; the poisoned sample is external on purpose.
SEED_TRUST = {"untrusted-vendor-note": TrustTier.external, "agent-rag-charter": TrustTier.verified}


def build_pipeline() -> RagPipeline:
    p = RagPipeline(generator=generator_from_env())
    if CORPUS.is_dir() and os.environ.get("AGENT_RAG_SEED", "1") == "1":
        p.index_directory(CORPUS, SEED_TRUST)
    return p


pipeline = build_pipeline()
app = FastAPI(title="Agent-Rag", version="0.1.0",
              description="Governed vector, hybrid and graph retrieval for AGenNext agents.")


class IndexResult(BaseModel):
    source_id: str
    chunks: int


@app.get("/healthz")
def healthz() -> dict:
    s = pipeline.store
    return {"status": "ok", "generator": pipeline.generator.name, "embedder": s.embedder.name,
            "sources": len(s.sources()), "chunks": len(s.chunks()), "entities": len(s.entities()),
            "relations": len(s.relations()), "communities": len(s.communities())}


@app.post("/v1/sources", response_model=IndexResult)
def upsert_source(doc: SourceDocument) -> IndexResult:
    """Stand-in for Agent-Knowledge sync: index one source document."""
    return IndexResult(source_id=doc.source_id, chunks=pipeline.index(doc))


@app.get("/v1/sources")
def list_sources() -> list[dict]:
    return [{"source_id": d.source_id, "title": d.title, "trust_tier": d.trust_tier, "uri": d.uri}
            for d in pipeline.store.sources().values()]


@app.delete("/v1/sources/{source_id}")
def delete_source(source_id: str) -> dict:
    if not pipeline.store.delete_source(source_id):
        raise HTTPException(404, "unknown source")
    return {"deleted": source_id}


@app.post("/v1/retrieve", response_model=RetrievalEnvelope)
def retrieve(req: RetrievalRequest) -> RetrievalEnvelope:
    return pipeline.retrieve(req)


@app.post("/v1/answer", response_model=GroundedAnswer)
def answer(req: RetrievalRequest) -> GroundedAnswer:
    return pipeline.answer(req)


@app.get("/v1/graph")
def graph(limit: int = Query(80, ge=1, le=1000)) -> dict:
    s = pipeline.store
    ents = sorted(s.entities().values(), key=lambda e: -e.mentions)[:limit]
    keep = {e.entity_id for e in ents}
    return {
        "entities": [e.model_dump(exclude={"chunk_ids"}) for e in ents],
        "relations": [r.model_dump(exclude={"chunk_ids"}) for r in s.relations().values() if r.source in keep and r.target in keep],
        "communities": [c.model_dump() for c in sorted(s.communities().values(), key=lambda c: -c.rank)],
    }


@app.get("/v1/traces", response_model=list[TraceEvent])
def traces(trace_id: str | None = None, limit: int = Query(200, ge=1, le=5000)) -> list[TraceEvent]:
    return pipeline.traces.events(trace_id, limit)


@app.get("/v1/traces/verify")
def verify_traces() -> dict:
    return pipeline.traces.verify()


@app.get("/")
def ui() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")
