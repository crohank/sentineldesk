"""Draft generation. The offline template generator is the default and always available.

Each draft line is a reviewed action whose evidence is a sentence in a retrieved
precedent's answer (`answer_snippet`), cited inline as `[TKT-12345]` (D5-14). The reply
is in the detected language even when the evidence is in the other language, because
every action has a reviewed English and German text.
"""
from __future__ import annotations

import os

from . import contracts as c
from .actions import ACTIONS, CLOSING, GREETING, LEAD, THANKS, Action
from .config import DEFAULT_CONFIG, SafetyConfig
from .gate import SUPPORTED_LANGUAGES, normalize_language
from .text import sentences

_ORDER = {a.id: i for i, a in enumerate(ACTIONS)}


def _evidence(p) -> str:
    return getattr(p, "answer_snippet", "") or ""


def _supporting_sentence(action: Action, precedent) -> str | None:
    for sentence in sentences(_evidence(precedent)):
        if action.supports(sentence):
            return sentence
    return None


def plan_actions(masked_ticket, precedents, *, config: SafetyConfig = DEFAULT_CONFIG):
    """Supported actions as (action, precedent, evidence sentence), at most `max_actions`, in catalogue order."""
    floor = config.evidence_threshold()
    eligible = sorted(
        (p for p in precedents or [] if float(getattr(p, "score", 0.0)) >= floor and _evidence(p).strip()),
        key=lambda p: -float(p.score),
    )
    ranked = []
    for action in ACTIONS:
        support = [(p, s) for p in eligible if (s := _supporting_sentence(action, p))]
        if support:
            weight = sum(float(p.score) for p, _ in support)
            best_precedent, span = support[0]  # eligible is sorted by score
            ranked.append((weight, action, best_precedent, span))
    ranked.sort(key=lambda r: (-r[0], _ORDER[r[1].id]))
    chosen = sorted(ranked[: config.max_actions], key=lambda r: _ORDER[r[1].id])
    return [(action, p, span) for _, action, p, span in chosen]


def draft_template(masked_ticket, language, precedents, *, config: SafetyConfig = DEFAULT_CONFIG) -> c.Draft:
    lang = normalize_language(language)
    plan = plan_actions(masked_ticket, precedents, config=config) if lang in SUPPORTED_LANGUAGES else []
    if not plan:
        return c.Draft(reply_text="", cited_ticket_ids=[], claims=[], abstain=True)
    lines = [GREETING[lang], "", THANKS[lang], LEAD[lang]]
    claims: list[c.Claim] = []
    cited: list[str] = []
    for action, p, span in plan:
        text = action.text(lang)
        lines.append(f"{text} [{p.ticket_id}]")
        claims.append(c.Claim(text=text, ticket_id=p.ticket_id, evidence_span=span))
        if p.ticket_id not in cited:
            cited.append(p.ticket_id)
    lines += ["", CLOSING[lang]]
    return c.Draft(reply_text="\n".join(lines), cited_ticket_ids=cited, claims=claims, abstain=False)


def draft_with_mode(masked_ticket, language, precedents, *, config: SafetyConfig = DEFAULT_CONFIG,
                    generator: str | None = None) -> tuple[c.Draft, str]:
    """Draft plus the generator that actually produced it.

    With generator "gemma", Gemma writes the draft; if it fails validation or the claim check
    after one repair, or the time budget runs out, the template draft is returned instead and
    the mode says offline_templates.
    """
    generator = generator or os.environ.get("SENTINELDESK_GENERATOR", c.OFFLINE_TEMPLATES)
    if generator == c.GEMMA and normalize_language(language) in SUPPORTED_LANGUAGES:
        from . import gemma
        from .validation import validate

        eligible = [p for p in precedents or [] if float(getattr(p, "score", 0.0)) >= config.evidence_threshold()]
        if eligible:
            result, _ = gemma.generate(
                masked_ticket, language, eligible[:5], budget_s=config.model_budget_s,
                validate_fn=lambda d: validate(d, language, precedents, ticket=masked_ticket, config=config),
            )
            if result is not None:
                return result, c.GEMMA
    return draft_template(masked_ticket, language, precedents, config=config), c.OFFLINE_TEMPLATES


def draft(masked_ticket, language, precedents) -> c.Draft:
    """`draft` stage for the pipeline (PLAN.md signature). SENTINELDESK_GENERATOR=gemma selects Gemma."""
    return draft_with_mode(masked_ticket, language, precedents)[0]
