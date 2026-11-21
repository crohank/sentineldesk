"""Mask personal data before logging, display or any external model call.

Tokens follow the KB's own placeholder kinds (D1-32), so triage and retrieval see the
tokens they were trained on: <email>, <tel_num>, <acc_num>. IP addresses become
<ip_address>. Ticket IDs, error codes, version numbers and dates are kept.

Known limit: personal names are not masked (no name recognizer).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..pipeline import MaskedTicket


@dataclass
class MaskResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)


_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
_CARD = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_ACCOUNT = re.compile(
    r"\b(?:account|acct|customer|client|contract|member|kunden|konto|vertrags)"
    r"[ -]?(?:number|no\.?|nr\.?|id|#|nummer)?\s*(?:is|lautet|ist)?\s*[:#]?\s*"
    r"(?P<id>(?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{4,})\b",
    re.IGNORECASE,
)
_IPV4 = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?![\d.])"
)
_IPV6 = re.compile(
    r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}(?![\w:])"
    r"|(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){1,6}:(?:[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4}){0,5})?(?![\w:])"
)
_PHONE = re.compile(r"(?<![\w-])(?:\+|00)?\(?\d[\d ()./-]{6,}\d(?![\w-])")
_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$|^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$")
_DOTTED_NUMBER = re.compile(r"^\d+(?:\.\d+){1,3}$")  # versions, amounts


def _luhn(digits: str) -> bool:
    total, double = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if double:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        double = not double
    return total % 10 == 0


def _is_card(candidate: str) -> bool:
    digits = re.sub(r"\D", "", candidate)
    return 13 <= len(digits) <= 19 and _luhn(digits)


def _is_phone(candidate: str) -> bool:
    c = candidate.strip()
    digits = re.sub(r"\D", "", c)
    return 8 <= len(digits) <= 15 and not _DATE.match(c) and not _DOTTED_NUMBER.match(c)


def _account_repl(m: re.Match) -> str:
    return m.group(0)[: m.start("id") - m.start()] + "<acc_num>"


def mask_pii(text: str | None) -> MaskResult:
    """Return the masked text and how many items of each kind were masked."""
    if not text:
        return MaskResult("", {})
    counts: dict[str, int] = {}

    def apply(kind: str, pattern: re.Pattern, s: str, token: str | None = None, check=None, repl=None) -> str:
        def _sub(m: re.Match) -> str:
            if check is not None and not check(m.group(0)):
                return m.group(0)
            counts[kind] = counts.get(kind, 0) + 1
            return repl(m) if repl is not None else token

        return pattern.sub(_sub, s)

    s = apply("email", _EMAIL, text, "<email>")
    s = apply("iban", _IBAN, s, "<acc_num>")
    s = apply("card", _CARD, s, "<acc_num>", check=_is_card)
    s = apply("account", _ACCOUNT, s, repl=_account_repl)
    s = apply("ip", _IPV4, s, "<ip_address>")
    s = apply("ip", _IPV6, s, "<ip_address>")
    s = apply("phone", _PHONE, s, "<tel_num>", check=_is_phone)
    return MaskResult(s, counts)


def mask_ticket(ticket) -> MaskedTicket:
    """`mask` stage for the pipeline: TicketInput -> MaskedTicket (subject may be missing)."""
    subject = mask_pii(getattr(ticket, "subject", None) or "").text
    body = mask_pii(getattr(ticket, "body", None) or "").text
    return MaskedTicket(subject=subject, body=body)
