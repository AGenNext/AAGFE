"""Agent-Rag contracts.

These are the wire-level shapes Agent-Rag exchanges with its neighbours
(Agent-Knowledge, Agent-Context, Agent-Guard, Agent-Cognitive-Guard,
Agent-Trust, Agent-Traces). Every other module depends on these, never the
other way round, so a component can be swapped without touching the rest.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Sources and knowledge units (owned upstream by Agent-Knowledge)
# --------------------------------------------------------------------------


class TrustTier(str, Enum):
    """Provenance tier of a source, as asserted by Agent-Knowledge / Agent-Trust."""

    verified = "verified"  # internal, signed or reviewed
    internal = "internal"  # internal, unreviewed
    external = "external"  # third party / web / vendor


class SourceDocument(BaseModel):
    source_id: str
    title: str
    text: str
    uri: str | None = None
    trust_tier: TrustTier = TrustTier.internal
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    title: str
    text: str
    ordinal: int
    trust_tier: TrustTier
    content_hash: str  # sha256 of text: provenance anchor for Agent-Trust


class Entity(BaseModel):
    entity_id: str
    name: str
    mentions: int = 0
    chunk_ids: list[str] = Field(default_factory=list)
    community_id: str | None = None


class Relation(BaseModel):
    source: str  # entity_id
    target: str  # entity_id
    weight: float
    chunk_ids: list[str] = Field(default_factory=list)


class Community(BaseModel):
    community_id: str
    entity_ids: list[str]
    title: str
    summary: str
    chunk_ids: list[str]
    summary_chunk_ids: list[str] = Field(default_factory=list)  # chunks the report quotes
    rank: float


# --------------------------------------------------------------------------
# Retrieval request / plan
# --------------------------------------------------------------------------


class RetrievalMode(str, Enum):
    auto = "auto"
    vector = "vector"
    lexical = "lexical"
    hybrid = "hybrid"
    graph_local = "graph_local"
    graph_global = "graph_global"


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    mode: RetrievalMode = RetrievalMode.auto
    top_k: int = Field(default=6, ge=1, le=50)
    agent_id: str = "agent:anonymous"
    team_id: str | None = None
    min_trust_tier: TrustTier | None = None  # drop sources below this tier


class RetrievalPlan(BaseModel):
    mode: RetrievalMode
    strategies: list[str]
    rationale: str
    matched_entities: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Evidence, guard verdicts and the result envelope
# --------------------------------------------------------------------------


class GuardDecision(str, Enum):
    allow = "allow"
    quarantine = "quarantine"
    block = "block"


class GuardFinding(BaseModel):
    guard: Literal["agent-guard", "agent-cognitive-guard"]
    rule: str
    decision: GuardDecision
    detail: str


class Evidence(BaseModel):
    """One retrieved unit plus everything needed to cite and audit it."""

    evidence_id: str
    kind: Literal["chunk", "community_report"]
    text: str
    title: str
    source_id: str | None
    chunk_id: str | None
    score: float
    retrievers: list[str]
    trust_tier: TrustTier
    content_hash: str
    decision: GuardDecision = GuardDecision.allow
    findings: list[GuardFinding] = Field(default_factory=list)


class ContextCandidate(BaseModel):
    """Input for Agent-Context. Explicitly NOT trusted context.

    Agent-Rag retrieves. Agent-Context decides what can be loaded.
    """

    trusted: Literal[False] = False
    handoff_to: Literal["Agent-Context"] = "Agent-Context"
    evidence_ids: list[str]
    token_estimate: int


class RetrievalEnvelope(BaseModel):
    envelope_id: str
    created_at: datetime = Field(default_factory=utcnow)
    request: RetrievalRequest
    plan: RetrievalPlan
    evidence: list[Evidence]
    quarantined: list[Evidence]
    blocked: list[Evidence]
    context_candidate: ContextCandidate
    trace_id: str
    timings_ms: dict[str, float]


# --------------------------------------------------------------------------
# Grounded generation (optional stage; model calls go through model-gateway)
# --------------------------------------------------------------------------


class Citation(BaseModel):
    marker: int
    evidence_id: str
    title: str
    source_id: str | None


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    generator: str
    groundedness: float  # share of answer sentences that carry a valid citation
    envelope: RetrievalEnvelope


# --------------------------------------------------------------------------
# Traces (Agent-Traces)
# --------------------------------------------------------------------------


class TraceEvent(BaseModel):
    seq: int
    trace_id: str
    at: datetime = Field(default_factory=utcnow)
    stage: str
    agent_id: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    prev_hash: str
    hash: str
