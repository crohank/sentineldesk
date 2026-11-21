"""Draft checks (PLAN.md, "Grounding contract"). Any error withholds the draft (rule 9).

1. Cited IDs are a non-empty subset of the retrieved precedents; inline markers are cited.
2. Every claim cites a cited ID, and its evidence span occurs in that precedent's answer.
3. Every sentence that isn't fixed boilerplate is a claim.
4. Numbers, currencies, units and qualifiers in a claim also appear in its evidence span.
5. The reply is in the requested language.
6. No mask placeholders in the reply.
7. No commitment terms (refund, guarantee, credit, ...) that the evidence doesn't contain.
8. At most `max_reply_words` words.
9. Model-written drafts only: no product-like names absent from the ticket and the evidence.
"""
from __future__ import annotations

import re

from . import contracts as c
from .actions import BOILERPLATE, TEMPLATE_TEXTS
from .config import DEFAULT_CONFIG, SafetyConfig
from .gate import SUPPORTED_LANGUAGES, normalize_language
from .patterns import COMMITMENT
from .text import CITATION, norm, sentences, strip_citations, ticket_text, word_count

_PLACEHOLDER = re.compile(r"<[A-Za-z][A-Za-z_ -]{0,30}>|\[(?!TKT-\d{5}\])[A-Za-z][A-Za-z _-]{0,30}\]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_UNIT = re.compile(
    r"[€$£%]|\b(?:eur|euro|usd|dollars?|percent|prozent|hours?|hrs?|days?|weeks?|months?|minutes?|mins?"
    r"|stunden?|tagen?|wochen?|monaten?|minuten?)\b",
    re.IGNORECASE,
)
_QUALIFIER = re.compile(
    r"\b(?:guarantee\w*|always|never|within|up\ to|at\ least|at\ most|only|immediately|instantly|garantier\w*"
    r"|immer|nie|niemals|innerhalb|bis\ zu|mindestens|h[oö]chstens|sofort|nur)\b",
    re.IGNORECASE | re.VERBOSE,
)
_PRODUCTISH = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9]*|[A-Za-z]+\d+[A-Za-z0-9]*|[A-Za-z]+\.[A-Za-z]{2,}|[a-z]+[A-Z][A-Za-z]*)\b")
_ALL_TEMPLATE_TEXTS = TEMPLATE_TEXTS["en"] | TEMPLATE_TEXTS["de"]


def _detect(text: str) -> str:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0
    try:
        return detect(text)
    except LangDetectException:
        return "und"


def is_template_draft(draft) -> bool:
    claims = list(getattr(draft, "claims", []) or [])
    return bool(claims) and all(norm(cl.text) in _ALL_TEMPLATE_TEXTS for cl in claims)


def validate(draft, language, precedents, *, ticket=None, config: SafetyConfig = DEFAULT_CONFIG) -> c.Validation:
    if getattr(draft, "abstain", False):
        return c.Validation(valid=False, errors=["draft abstained"])
    errors: list[str] = []
    by_id = {p.ticket_id: p for p in precedents or []}
    cited = list(getattr(draft, "cited_ticket_ids", []) or [])
    claims = list(getattr(draft, "claims", []) or [])
    reply = getattr(draft, "reply_text", "") or ""
    body = strip_citations(reply)
    lang = normalize_language(language)
    template = is_template_draft(draft)

    # 1 citations
    if not cited:
        errors.append("no citations")
    for tid in cited:
        if tid not in by_id:
            errors.append(f"cited {tid} was not retrieved")
    for tid in CITATION.findall(reply):
        if tid not in cited:
            errors.append(f"inline citation {tid} is missing from cited_ticket_ids")

    # 2 claims and evidence spans
    if not claims:
        errors.append("no claims")
    span_for: dict[str, str] = {}
    for i, claim in enumerate(claims):
        if claim.ticket_id not in cited:
            errors.append(f"claim {i} cites {claim.ticket_id}, which is not in cited_ticket_ids")
        p = by_id.get(claim.ticket_id)
        if p is not None and (not norm(claim.evidence_span)
                              or norm(claim.evidence_span) not in norm(getattr(p, "answer_snippet", ""))):
            errors.append(f"claim {i} evidence span not found in {claim.ticket_id}")
        span_for[norm(claim.text)] = claim.evidence_span

    # 3, 4, 7 sentence support, numbers, units, qualifiers, commitments
    boilerplate = BOILERPLATE.get(lang, set())
    for sentence in sentences(body):
        key = norm(sentence)
        if key in boilerplate:
            continue
        span = span_for.get(key)
        if span is None:
            errors.append(f"unsupported sentence: {sentence[:60]}")
            continue
        low = span.lower()
        errors += [f"number not in evidence: {n}" for n in _NUMBER.findall(sentence) if n not in span]
        errors += [f"unit not in evidence: {u}" for u in _UNIT.findall(sentence) if u.lower() not in low]
        errors += [f"qualifier not in evidence: {q}" for q in _QUALIFIER.findall(sentence) if q.lower() not in low]
        errors += [f"commitment not in evidence: {m.group(0)}" for m in COMMITMENT.finditer(sentence)
                   if m.group(0).lower() not in low]

    # 5 language
    if lang not in SUPPORTED_LANGUAGES:
        errors.append(f"unsupported reply language: {lang}")
    elif template:
        wrong = [cl.text for cl in claims if norm(cl.text) not in TEMPLATE_TEXTS[lang]]
        if wrong:
            errors.append(f"template text is not the reviewed {lang} version")
    else:
        detected = _detect(body)
        if detected != lang:
            errors.append(f"reply language {detected} does not match {lang}")

    # 6 placeholders
    m = _PLACEHOLDER.search(body)
    if m:
        errors.append(f"placeholder in reply: {m.group(0)}")

    # 8 length
    words = word_count(body)
    if words > config.max_reply_words:
        errors.append(f"reply too long: {words} words")

    # 9 product-like names in model-written drafts
    if not template:
        known = (ticket_text(ticket) + " " + " ".join(getattr(p, "answer_snippet", "") for p in by_id.values())).lower()
        unknown = sorted({t for t in _PRODUCTISH.findall(body) if t.lower() not in known})
        if unknown:
            errors.append(f"names not in ticket or evidence: {', '.join(unknown[:5])}")

    return c.Validation(valid=not errors, errors=errors)
