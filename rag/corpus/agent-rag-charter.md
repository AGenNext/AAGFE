# Agent-Rag

Agent-Rag owns retrieval-augmented generation for AGenNext, including vector RAG, hybrid RAG, and graph-based RAG.

## Decision

Agent-Rag provides governed retrieval pipelines for agents and agent teams.

It supports Microsoft GraphRAG-inspired graph retrieval patterns while remaining native to the AGenNext architecture: SurrealDB, Agent-Knowledge, Agent-Context, Agent-Guard, Agent-Cognitive-Guard, Agent-Trust, Agent-Traces, and Agent-Platform.

This repository does not copy Microsoft GraphRAG code. It implements AGenNext-compatible graph RAG contracts and runtime stubs.

## Scope

Agent-Rag owns:

- retrieval pipelines
- vector retrieval
- hybrid retrieval
- graph retrieval
- entity/claim/community retrieval contracts
- retrieval planning
- citation/evidence retrieval
- retrieval result envelopes
- context assembly inputs for Agent-Context
- RAG safety handoff to Agent-Guard and Agent-Cognitive-Guard

Agent-Rag does not own:

- raw source ingestion
- long-term knowledge storage
- final trusted context loading
- platform authority
- model routing
- user-facing artifact generation

## Boundary

| Component | Responsibility |
|---|---|
| Agent-Rag | Retrieval pipelines and graph RAG orchestration |
| Agent-Knowledge | Lossless source storage and structured knowledge extraction |
| Agent-Context | Controls trusted context loading |
| Agent-Guard | Screens external/poisoned retrieval content |
| Agent-Cognitive-Guard | Screens context/belief/mind poisoning risks |
| Agent-Trust | Provenance and evidence contracts |
| Agent-Traces | Retrieval lifecycle evidence |
| model-router | Selects retrieval/generation-capable models |
| model-gateway | Executes model calls |

## GraphRAG-inspired pipeline

```txt
sources in Agent-Knowledge
  ↓
entities / claims / relations / communities
  ↓
SurrealDB graph + vector indexes
  ↓
query planning
  ↓
local retrieval + global/community retrieval
  ↓
Guard and Cognitive Guard checks
  ↓
trusted context candidate for Agent-Context
  ↓
grounded generation/evaluation
```

## Rule

Retrieval output is not automatically trusted context.

Agent-Rag retrieves. Agent-Context decides what can be loaded. Guard layers decide what must be blocked or quarantined.
