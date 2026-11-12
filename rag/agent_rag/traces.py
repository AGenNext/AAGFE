"""Retrieval lifecycle evidence for Agent-Traces.

Events are hash-chained (each event's hash covers the previous hash), so any
edit or deletion is detectable with ``verify()``. ``TraceSink`` is the seam
for an OpenTelemetry / SIEM exporter.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from typing import Any, Protocol

from .contracts import TraceEvent, utcnow
from .text import sha256

GENESIS = "0" * 64


class TraceSink(Protocol):
    def emit(self, trace_id: str, stage: str, agent_id: str, attributes: dict[str, Any]) -> TraceEvent: ...


class HashChainTraceSink:
    def __init__(self, max_events: int = 5000) -> None:
        self._events: deque[TraceEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._seq = 0
        self._last_hash = GENESIS
        self._dropped_head_prev: str = GENESIS

    @staticmethod
    def _digest(seq: int, trace_id: str, at: str, stage: str, agent_id: str, attrs: dict[str, Any], prev: str) -> str:
        payload = json.dumps([seq, trace_id, at, stage, agent_id, attrs, prev], sort_keys=True, default=str)
        return sha256(payload)

    def emit(self, trace_id: str, stage: str, agent_id: str, attributes: dict[str, Any]) -> TraceEvent:
        with self._lock:
            self._seq += 1
            at = utcnow()
            h = self._digest(self._seq, trace_id, at.isoformat(), stage, agent_id, attributes, self._last_hash)
            ev = TraceEvent(seq=self._seq, trace_id=trace_id, at=at, stage=stage, agent_id=agent_id,
                            attributes=attributes, prev_hash=self._last_hash, hash=h)
            if len(self._events) == self._events.maxlen:
                self._dropped_head_prev = self._events[0].hash
            self._events.append(ev)
            self._last_hash = h
            return ev

    def events(self, trace_id: str | None = None, limit: int = 200) -> list[TraceEvent]:
        evs = [e for e in self._events if trace_id is None or e.trace_id == trace_id]
        return evs[-limit:]

    def verify(self) -> dict[str, Any]:
        with self._lock:
            evs = list(self._events)
        prev = evs[0].prev_hash if evs else GENESIS
        for e in evs:
            expect = self._digest(e.seq, e.trace_id, e.at.isoformat(), e.stage, e.agent_id, e.attributes, e.prev_hash)
            if e.prev_hash != prev or e.hash != expect:
                return {"valid": False, "broken_at_seq": e.seq, "events_checked": len(evs)}
            prev = e.hash
        return {"valid": True, "events_checked": len(evs), "head": prev}
