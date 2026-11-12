# Agent-Rag service

A working implementation of the [Agent-Rag charter](https://github.com/AGenNext/agent-rag): governed vector, hybrid and GraphRAG-style retrieval for AGenNext agents.

> **Rule:** retrieval output is not automatically trusted context.
> Agent-Rag retrieves. Agent-Context decides what can be loaded. Guard layers decide what must be blocked or quarantined.

![Agent-Rag console](docs/console.png)

## Quick start

```bash
cd rag
pip install -e ".[dev]"
uvicorn agent_rag.api:app --port 8088
```

Open `http://localhost:8088`. The service indexes `corpus/` on startup: the AGenNext and AAGFE docs, the Agent-Rag charter (`verified`), and a deliberately poisoned vendor note (`external`) that shows the guards working.

```bash
python -m pytest        # 23 tests
```

## Pipeline

```txt
sources (stand-in for Agent-Knowledge sync)       POST /v1/sources
  ↓  markdown-aware chunking + sha256 per chunk    provenance anchor for Agent-Trust
  ↓  entities / relations / communities            graph.py (label propagation, extractive reports)
  ↓  vector + BM25 + graph indexes                 store.py  (IndexStore protocol)
query planning                                     retrieval.QueryPlanner
  ↓  local: entity match → 1-hop expansion → RRF with hybrid
  ↓  global: community reports + grounding chunks
  ↓  hybrid: vector + BM25 → reciprocal rank fusion
trust-tier filter (min_trust_tier)
  ↓
Agent-Guard + Agent-Cognitive-Guard                guards.py → allow / quarantine / block
  ↓
context candidate {trusted: false, handoff_to: Agent-Context}
  ↓
optional grounded generation + groundedness score  generation.py (model-gateway seam)
every stage → hash-chained Agent-Traces            traces.py, GET /v1/traces/verify
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/retrieve` | `RetrievalRequest` → `RetrievalEnvelope` (plan, allowed / quarantined / blocked evidence, context candidate, trace id) |
| `POST` | `/v1/answer` | Same as retrieve, plus a cited answer and its groundedness score |
| `POST` / `GET` / `DELETE` | `/v1/sources[/{id}]` | Index, list or remove a source document (stand-in for Agent-Knowledge sync) |
| `GET` | `/v1/graph` | Entities, relations, community reports |
| `GET` | `/v1/traces`, `/v1/traces/verify` | Retrieval lifecycle events and hash-chain verification |
| `GET` | `/healthz` | Index stats, active generator and embedder |

OpenAPI docs: `http://localhost:8088/docs`.

```bash
curl -s localhost:8088/v1/retrieve -H 'content-type: application/json' \
  -d '{"query":"What does Agent-Guard do?","mode":"auto","top_k":6,"agent_id":"agent:demo"}'
```

`mode` is one of `auto`, `vector`, `lexical`, `hybrid`, `graph_local` or `graph_global`. `min_trust_tier` can be `external`, `internal` or `verified`.

## Guards (reference implementations)

| Guard | Rule | Decision |
|---|---|---|
| Agent-Guard | `prompt_injection`, `secret_exfiltration`, `role_hijack`, `embedded_secret` | block |
| Agent-Cognitive-Guard | `belief_override`, `trust_escalation` (negation-aware) | quarantine |
| Agent-Cognitive-Guard | `conflicting_claim`: the same statement with opposite polarity to a higher-trust source | quarantine |
| Agent-Cognitive-Guard | `external_source`: external provenance with no other findings | allow (flagged) |

Community reports only quote non-external chunks, so poisoned third-party text can't get into a graph summary. A report's trust tier is the weakest tier among the chunks it quotes.

## Composability: swap points

| Seam | Default | Replace with |
|---|---|---|
| `IndexStore` | `InMemoryIndexStore` | SurrealDB graph + vector adapter |
| `Embedder` | `HashingEmbedder` (offline; lexical, not semantic) | An embedding model behind model-gateway |
| `EntityExtractor` | Heuristic proper-noun / product-name extractor | LLM extraction via model-gateway |
| `Guard` | Local rule-based guards | Remote Agent-Guard / Agent-Cognitive-Guard clients |
| `Generator` | `ExtractiveGenerator` (no model call) | `ClaudeGenerator`: `AGENT_RAG_GENERATOR=claude`, optional `ANTHROPIC_BASE_URL` to route through model-gateway |
| `TraceSink` | In-memory hash chain | OpenTelemetry / SIEM exporter |

## Known limitations

- The default embedder is feature hashing, so recall is lexical. Paraphrased questions need a real embedding model.
- The guards use pattern heuristics. They catch the demo attacks and are tested against a negation false positive, but they are not a complete defence. Production should call the real Guard services.
- `conflicting_claim` compares sentence polarity with token overlap (Jaccard ≥ 0.6). Contradictions that use different wording are missed.
- The index and traces are in memory and are lost on restart.
- `ClaudeGenerator` hasn't been run against the live API in this repository: no credentials were available when it was built. The extractive generator is the tested default.
