"""Small text helpers shared by the gate, drafting and validation."""
from __future__ import annotations

import re

CITATION = re.compile(r"\s*\[(TKT-\d{5})\]")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"\w+")


def sentences(text: str | None) -> list[str]:
    """Split on sentence punctuation and line breaks. Each piece is a substring of `text`."""
    return [s.strip() for s in _SENTENCE_BREAK.split(text or "") if s and s.strip()]


def norm(text: str | None) -> str:
    """Collapse whitespace, for comparisons only."""
    return " ".join((text or "").split())


def strip_citations(text: str | None) -> str:
    return CITATION.sub("", text or "")


def word_count(text: str | None) -> int:
    return len(_WORD.findall(text or ""))


def ticket_text(ticket) -> str:
    subject = getattr(ticket, "subject", "") or ""
    body = getattr(ticket, "body", "") or ""
    return f"{subject}\n{body}".strip()
