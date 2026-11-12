"""RAG safety handoff to Agent-Guard and Agent-Cognitive-Guard.

Agent-Rag does not own the guard decision logic; it owns the *handoff*.
``Guard`` is the protocol a remote Agent-Guard / Agent-Cognitive-Guard client
implements. The local rule-based guards below are reference implementations
so the pipeline is safe-by-default when no remote guard is configured. They
are pattern heuristics, not a complete defence.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from .contracts import Evidence, GuardDecision, GuardFinding, TrustTier
from .store import IndexStore
from .text import sentences, tokenize

_SEVERITY = {GuardDecision.allow: 0, GuardDecision.quarantine: 1, GuardDecision.block: 2}


def worst(decisions: Sequence[GuardDecision]) -> GuardDecision:
    return max(decisions, key=_SEVERITY.__getitem__, default=GuardDecision.allow)


class Guard(Protocol):
    name: str

    def screen(self, query: str, evidence: list[Evidence]) -> None:
        """Append findings to each evidence item in place."""


class AgentGuard:
    """Screens retrieved content for adversarial / poisoned input."""

    name = "agent-guard"

    RULES: list[tuple[str, re.Pattern[str], GuardDecision, str]] = [
        (
            "prompt_injection",
            re.compile(r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|all)\b.{0,20}\b(instructions?|rules?|prompts?)\b", re.I),
            GuardDecision.block,
            "instruction-override phrasing inside retrieved content",
        ),
        (
            "secret_exfiltration",
            re.compile(r"\b(reveal|print|dump|exfiltrate|send|leak)\b.{0,40}\b(secrets?|passwords?|api[- ]?keys?|tokens?|credentials?)\b", re.I),
            GuardDecision.block,
            "request to disclose secrets embedded in content",
        ),
        (
            "role_hijack",
            re.compile(r"\b(you are now|act as|new system prompt|system prompt:)\b", re.I),
            GuardDecision.block,
            "attempt to redefine the agent's role",
        ),
        (
            "embedded_secret",
            re.compile(r"(AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|\bsk-[A-Za-z0-9_\-]{20,})"),
            GuardDecision.block,
            "credential material present in retrieved content",
        ),
    ]

    def screen(self, query: str, evidence: list[Evidence]) -> None:
        for ev in evidence:
            for rule, pat, decision, detail in self.RULES:
                if pat.search(ev.text):
                    ev.findings.append(GuardFinding(guard="agent-guard", rule=rule, decision=decision, detail=detail))


_NEG = re.compile(r"\b(not|never|no|cannot|can't|doesn't|isn't|aren't|won't|without)\b|n't\b", re.I)


def _affirmed(pattern: re.Pattern[str], sentence: str) -> bool:
    """True if ``pattern`` matches without a negation in the 3 words before it."""
    for m in pattern.finditer(sentence):
        before = sentence[: m.start()].split()[-3:]
        if not _NEG.search(" ".join(before)):
            return True
    return False


_TIER_RANK = {TrustTier.verified: 2, TrustTier.internal: 1, TrustTier.external: 0}


class CognitiveGuard:
    """Screens context / belief poisoning risks.

    * ``belief_override`` – content tries to install a persistent belief.
    * ``trust_escalation`` – content claims it should bypass review / be trusted.
    * ``conflicting_claim`` – content negates a claim made by a more trusted
      source (same statement, opposite polarity).
    """

    name = "agent-cognitive-guard"

    BELIEF = re.compile(r"\b(from now on|always believe|you must (always )?(believe|assume)|remember that you)\b", re.I)
    ESCALATION = re.compile(
        r"\b(automatically trusted|does not need [\w\- ]{0,30}(review|approval)|bypass(es)? [\w\- ]{0,20}(review|approval|guard|policy)|skip (the )?(guard|review))\b",
        re.I,
    )

    def __init__(self, store: IndexStore, *, jaccard_threshold: float = 0.6) -> None:
        self.store = store
        self.threshold = jaccard_threshold

    def _reference_claims(self, below: TrustTier) -> list[tuple[set[str], bool, str]]:
        out = []
        for ch in self.store.chunks().values():
            if _TIER_RANK[ch.trust_tier] <= _TIER_RANK[below]:
                continue
            for s in sentences(ch.text):
                toks = set(tokenize(s))
                if len(toks) >= 3:
                    out.append((toks, bool(_NEG.search(s)), ch.source_id))
        return out

    def screen(self, query: str, evidence: list[Evidence]) -> None:
        for ev in evidence:
            sents = sentences(ev.text)
            # A match negated just before it ("is not automatically trusted")
            # restates the rule rather than attacking it, so it doesn't count.
            if any(_affirmed(self.BELIEF, s) for s in sents):
                ev.findings.append(
                    GuardFinding(guard="agent-cognitive-guard", rule="belief_override", decision=GuardDecision.quarantine,
                                 detail="content attempts to install a persistent belief")
                )
            if any(_affirmed(self.ESCALATION, s) for s in sents):
                ev.findings.append(
                    GuardFinding(guard="agent-cognitive-guard", rule="trust_escalation", decision=GuardDecision.quarantine,
                                 detail="content claims elevated trust or review bypass")
                )
            refs = self._reference_claims(ev.trust_tier)
            for s in sents:
                toks = set(tokenize(s))
                if len(toks) < 3:
                    continue
                neg = bool(_NEG.search(s))
                for rtoks, rneg, src in refs:
                    if neg != rneg and len(toks & rtoks) / len(toks | rtoks) >= self.threshold:
                        ev.findings.append(
                            GuardFinding(guard="agent-cognitive-guard", rule="conflicting_claim", decision=GuardDecision.quarantine,
                                         detail=f"contradicts a higher-trust claim in {src}: '{s[:90]}'")
                        )
                        break
            if ev.trust_tier is TrustTier.external and not ev.findings:
                ev.findings.append(
                    GuardFinding(guard="agent-cognitive-guard", rule="external_source", decision=GuardDecision.allow,
                                 detail="external provenance; allowed but flagged for Agent-Context weighting")
                )


def apply_guards(guards: Sequence[Guard], query: str, evidence: list[Evidence]) -> None:
    for g in guards:
        g.screen(query, evidence)
    for ev in evidence:
        ev.decision = worst([f.decision for f in ev.findings])
