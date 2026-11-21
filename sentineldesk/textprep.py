"""Text cleanup shared by indexing and live queries (D3-04).

The same function must run on KB documents and on incoming tickets, otherwise
query and document vectors drift apart.
"""
from __future__ import annotations

import re

# Leading salutation: "Dear Customer Support Team,", "Hello,", "Sehr geehrter Kundenservice,", "<name>,"
_SALUTATION = re.compile(
    r"^\s*(?:"
    r"(?:dear|hello|hi|hey|greetings|good (?:morning|afternoon|evening))\b[^,\n.!]{0,50}[,!.:]?"
    r"|(?:sehr geehrte[rs]?(?:/r)?|liebe[rs]?|hallo|guten (?:tag|morgen|abend))\b[^,\n.!]{0,50}[,!.:]?"
    r"|(?:customer support|support team|kundenservice|kundendienst)[^,\n.!]{0,20},"
    r"|(?:<[^<>]{1,20}>|\[[^\[\]]{1,20}\])\s*,"
    r")\s*",
    re.IGNORECASE,
)

# Trailing courtesy sentences, removed repeatedly from the end.
_CLOSER = re.compile(
    r"(?:"
    r"thank(?:s| you)\b[^.!?\n]{0,80}[.!]?"
    r"|(?:i|we) (?:look forward|am looking forward|would (?:greatly )?appreciate)[^.!?\n]{0,120}[.!]?"
    r"|looking forward[^.!?\n]{0,80}[.!]?"
    r"|(?:best|kind|warm)? ?regards[^.!?\n]{0,40}[.,!]?"
    r"|sincerely[^.!?\n]{0,40}[.,]?"
    r"|vielen dank[^.!?\n]{0,80}[.!]?"
    r"|danke[^.!?\n]{0,80}[.!]?"
    r"|ich freue mich[^.!?\n]{0,100}[.!]?"
    r"|wir freuen uns[^.!?\n]{0,100}[.!]?"
    r"|mit freundlichen gr(?:ü|ue)(?:ß|ss)en[^.!?\n]{0,40}[.,]?"
    r"|(?:beste|viele) gr(?:ü|ue)(?:ß|ss)e[^.!?\n]{0,40}[.,]?"
    r"|<[^<>]{1,30}>|\[[^\[\]]{1,30}\]"
    r")\s*$",
    re.IGNORECASE,
)

_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_MASK = re.compile(r"<[^<>]{1,30}>|\[[^\[\]]{1,30}\]")
_WS = re.compile(r"\s+")


def normalize(text: str | None) -> str:
    """Decode literal \\n, drop <br>, collapse whitespace."""
    if not text:
        return ""
    text = text.replace("\\n", "\n")
    text = _BR.sub("\n", text)
    return text.strip()


def strip_boilerplate(text: str) -> str:
    text = _SALUTATION.sub("", text, count=1)
    for _ in range(4):
        new = _CLOSER.sub("", text.rstrip())
        if new == text:
            break
        text = new
    return text.strip()


def clean_for_search(subject: str | None, body: str | None) -> str:
    """Search text for a ticket: subject + body without boilerplate or mask tokens."""
    body_clean = strip_boilerplate(normalize(body))
    text = f"{normalize(subject)}\n{body_clean}"
    text = _MASK.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def answer_snippet(answer: str, max_chars: int = 300) -> str:
    """Readable answer excerpt for the console: no salutation, cut at a word boundary."""
    text = _SALUTATION.sub("", normalize(answer), count=1)
    text = _WS.sub(" ", text).strip()
    text = text[:1].upper() + text[1:]
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + " …"
