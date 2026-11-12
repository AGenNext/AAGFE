"""Text utilities: tokenisation, sentence splitting, chunking, hashing."""

from __future__ import annotations

import hashlib
import re

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-_]*")

STOPWORDS = frozenset(
    """a an and are as at be but by can could did do does for from has have how i if in
    into is it its may might must not of on or our should so such than that the their them
    then there these they this to was we were what when where which who why will with would
    you your all any also each other only own same too very just about over under between""".split()
)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return toks if keep_stopwords else [t for t in toks if t not in STOPWORDS]


HEADING_MARK = "§ "


def is_heading(sentence: str) -> bool:
    return sentence.startswith(HEADING_MARK.strip())


def normalize_markdown(text: str) -> list[str]:
    """Turn markdown into clean statement lines.

    Headings lose their ``#``, bullets their marker, table rows become
    ``cell: cell`` statements, fences and separator rows are dropped. Keeping
    one statement per line is what lets lists and tables be cited precisely.
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("```") or re.fullmatch(r"\|?[\s:\-|]+\|?", line):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            line = ": ".join(cells)
        heading = line.startswith("#")
        line = re.sub(r"^(#+|[-*+]|\d+\.)\s+", "", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        if line:
            # Headings stay in the chunk (they help retrieval) but carry a
            # marker so generators never present them as claims.
            out.append(f"{HEADING_MARK}{line}" if heading else line)
    return out


def sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in text.split("\n"):
        parts.extend(p.strip(" \t-*#|") if not p.startswith("§") else p.strip() for p in re.split(r"(?<=[.!?])\s+", line))
    return [p for p in parts if len(p) > 2]


def chunk_text(text: str, *, max_words: int = 120, overlap_words: int = 20) -> list[str]:
    """Line-aware chunking with overlap.

    Statement lines (see ``normalize_markdown``) are packed until
    ``max_words``; the tail lines of a chunk (up to ``overlap_words``) are
    repeated at the start of the next one so claims near a boundary stay
    retrievable from either side. Oversized lines are split on a word window.
    """
    lines: list[str] = []
    for ln in normalize_markdown(text):
        words = ln.split()
        if len(words) <= max_words:
            lines.append(ln)
            continue
        step = max(1, max_words - overlap_words)
        for i in range(0, len(words), step):
            lines.append(" ".join(words[i : i + max_words]))
            if i + max_words >= len(words):
                break

    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for ln in lines:
        n = len(ln.split())
        if buf and size + n > max_words:
            chunks.append("\n".join(buf))
            tail: list[str] = []
            t = 0
            for prev in reversed(buf):
                w = len(prev.split())
                if t + w > overlap_words:
                    break
                tail.insert(0, prev)
                t += w
            buf, size = tail, t
        buf.append(ln)
        size += n
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def estimate_tokens(text: str) -> int:
    # Rough, model-agnostic estimate (~0.75 words per token). Agent-Context
    # owns the real budget; this is only a sizing hint.
    return max(1, round(len(text.split()) / 0.75))
