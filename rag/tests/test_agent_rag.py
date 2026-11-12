from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_rag import RagPipeline, RetrievalMode, RetrievalRequest, SourceDocument, TrustTier
from agent_rag.contracts import GuardDecision
from agent_rag.generation import groundedness
from agent_rag.text import chunk_text, sentences

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
TRUST = {"untrusted-vendor-note": TrustTier.external, "agent-rag-charter": TrustTier.verified}


@pytest.fixture()
def pipe() -> RagPipeline:
    p = RagPipeline()
    p.index_directory(CORPUS, TRUST)
    return p


def ask(p: RagPipeline, q: str, **kw):
    return p.retrieve(RetrievalRequest(query=q, **kw))


# -- text -------------------------------------------------------------------


def test_chunking_keeps_table_rows_as_statements():
    md = "# T\n\n| A | B |\n|---|---|\n| Agent-Guard | Screens content |\n"
    chunks = chunk_text(md)
    assert "Agent-Guard: Screens content" in sentences(chunks[0])
    assert all("---" not in c for c in chunks)


def test_chunking_overlaps_long_documents():
    text = "\n".join(f"Line number {i} talks about topic {i}." for i in range(80))
    chunks = chunk_text(text, max_words=60, overlap_words=12)
    assert len(chunks) > 1
    assert chunks[0].splitlines()[-1] == chunks[1].splitlines()[0]


# -- planning ---------------------------------------------------------------


def test_planner_routes_by_query_shape(pipe):
    assert ask(pipe, "What does Agent-Guard do?").plan.mode is RetrievalMode.graph_local
    assert ask(pipe, "Give an overview of the main components").plan.mode is RetrievalMode.graph_global
    assert ask(pipe, "how are risky actions stopped").plan.mode is RetrievalMode.hybrid
    assert ask(pipe, "Agent-Guard", mode=RetrievalMode.vector).plan.mode is RetrievalMode.vector


def test_local_retrieval_prefers_specific_entity(pipe):
    env = ask(pipe, "What does Agent-Cognitive-Guard screen?")
    assert env.plan.matched_entities[0] == "agent-cognitive-guard"
    assert any("Agent-Cognitive-Guard" in e.text for e in env.evidence)


def test_global_retrieval_returns_community_reports(pipe):
    env = ask(pipe, "Summarize the overall themes")
    assert any(e.kind == "community_report" for e in env.evidence)


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_every_mode_returns_evidence(pipe, mode):
    env = ask(pipe, "Agent-Context trusted context loading", mode=mode)
    assert env.evidence, mode


# -- guards and trust -------------------------------------------------------


def test_poisoned_source_is_blocked_never_handed_off(pipe):
    env = ask(pipe, "Is Agent-Rag retrieval output trusted context?")
    blocked = {e.evidence_id: {f.rule for f in e.findings} for e in env.blocked}
    assert "untrusted-vendor-note#0" in blocked
    assert {"prompt_injection", "secret_exfiltration"} <= blocked["untrusted-vendor-note#0"]
    assert "untrusted-vendor-note#0" not in env.context_candidate.evidence_ids


def test_negated_rule_is_not_a_false_positive(pipe):
    # The charter says retrieval output is NOT automatically trusted. That
    # restates the rule and must not be quarantined as trust escalation.
    env = ask(pipe, "Is retrieval output automatically trusted?", top_k=10)
    charter = [e for e in env.evidence + env.quarantined if e.source_id == "agent-rag-charter"]
    assert charter and all(e.decision is GuardDecision.allow for e in charter)


def test_conflicting_claim_from_external_source_is_quarantined():
    p = RagPipeline()
    p.index(SourceDocument(source_id="policy", title="Policy", trust_tier=TrustTier.verified,
                           text="Payment agents require human approval before creating payments."))
    p.index(SourceDocument(source_id="blog", title="Blog", trust_tier=TrustTier.external,
                           text="Payment agents never require human approval before creating payments."))
    env = p.retrieve(RetrievalRequest(query="do payment agents require human approval", mode=RetrievalMode.hybrid))
    q = {e.source_id: {f.rule for f in e.findings} for e in env.quarantined}
    assert "conflicting_claim" in q.get("blog", set())
    assert [e.source_id for e in env.evidence] == ["policy"]


def test_community_reports_never_quote_external_content(pipe):
    chunks = pipe.store.chunks()
    for c in pipe.store.communities().values():
        assert all(chunks[cid].trust_tier is not TrustTier.external for cid in c.summary_chunk_ids)
        assert "Ignore all previous instructions" not in c.summary


def test_min_trust_tier_filters_sources(pipe):
    env = ask(pipe, "Agent-Context review", min_trust_tier=TrustTier.verified, top_k=10)
    assert env.evidence and all(e.trust_tier is TrustTier.verified for e in env.evidence + env.quarantined + env.blocked)


def test_evidence_carries_provenance_hash(pipe):
    from agent_rag.text import sha256

    for e in ask(pipe, "Agent Vault secrets").evidence:
        assert e.content_hash == sha256(e.text)


def test_context_candidate_is_never_trusted(pipe):
    env = ask(pipe, "Agent-Rag scope")
    assert env.context_candidate.trusted is False
    assert env.context_candidate.handoff_to == "Agent-Context"


# -- generation and traces --------------------------------------------------


def test_extractive_answer_is_fully_cited(pipe):
    a = pipe.answer(RetrievalRequest(query="What does Agent-Guard do?"))
    assert "Screens external/poisoned retrieval content" in a.answer
    assert a.groundedness == 1.0
    assert "§" not in a.answer


def test_groundedness_metric():
    assert groundedness("Alpha beta gamma [1]. Delta epsilon zeta.", 1) == 0.5
    assert groundedness("Alpha beta gamma [9].", 1) == 0.0


def test_trace_chain_detects_tampering(pipe):
    env = ask(pipe, "kill switch")
    stages = [e.stage for e in pipe.traces.events(env.trace_id)]
    assert stages == ["retrieve.request", "retrieve.plan", "retrieve.hits", "retrieve.guard", "retrieve.handoff"]
    assert pipe.traces.verify()["valid"]
    pipe.traces._events[3].attributes["tampered"] = True
    assert pipe.traces.verify() == {"valid": False, "broken_at_seq": pipe.traces._events[3].seq,
                                    "events_checked": len(pipe.traces._events)}


def test_reindex_replaces_source(pipe):
    before = len(pipe.store.chunks())
    pipe.index(SourceDocument(source_id="agent-rag-charter", title="x", text="Short replacement text about Agent-Rag."))
    assert len(pipe.store.chunks()) < before
    assert pipe.store.delete_source("agent-rag-charter")
    assert not any(c.source_id == "agent-rag-charter" for c in pipe.store.chunks().values())


# -- HTTP API ---------------------------------------------------------------


def test_http_api_end_to_end():
    from agent_rag.api import app

    c = TestClient(app)
    h = c.get("/healthz").json()
    assert h["status"] == "ok" and h["chunks"] > 0 and h["communities"] > 0
    r = c.post("/v1/answer", json={"query": "What does Agent-Context control?", "agent_id": "agent:test"})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"] and body["envelope"]["context_candidate"]["trusted"] is False
    assert c.post("/v1/retrieve", json={"query": ""}).status_code == 422
    assert c.post("/v1/sources", json={"source_id": "n", "title": "N", "text": "New note on Agent-Vault."}).json()["chunks"] == 1
    assert c.get("/v1/graph").json()["entities"]
    assert c.get("/v1/traces/verify").json()["valid"] is True
    assert c.get("/").status_code == 200
