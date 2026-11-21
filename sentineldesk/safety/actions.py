"""Reviewed support actions for the offline template generator.

An action may appear in a draft only if one sentence of a retrieved precedent's answer
matches all of its evidence patterns. That sentence becomes the claim's evidence span.
Texts are fixed and reviewed in English and German, and contain no numbers,
commitments or product names.

Only information requests are in the catalogue. Troubleshooting steps (restart, update,
clear the cache, check the connection) were tried and left out: in the KB answers,
sentences that mention them mostly report what the customer already tried or ask for
details about it, so a recommended step would often be unsupported. Support rates and
examples are in reports/d4/action_mining.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .text import norm

_F = re.IGNORECASE | re.VERBOSE

# A request to the customer: "please provide", "could you send", "bitte teilen Sie".
ASK = re.compile(
    r"""
    \b(?:provide|share|send|attach|forward|include|let\ us\ know|tell\ us|could\ you|can\ you|would\ you|please
    |confirm|specify|describe|bitte|senden|schicken|teilen|[uü]bermitteln|zukommen|mitteilen|geben\ sie|nennen
    |angeben|beschreiben|k[oö]nnten\ sie|k[oö]nnen\ sie)\b
    """,
    _F,
)


@dataclass(frozen=True)
class Action:
    id: str
    en: str
    de: str
    evidence: tuple[re.Pattern, ...]  # all must match one sentence of a precedent's answer

    def text(self, language: str) -> str:
        return self.de if language == "de" else self.en

    def supports(self, sentence: str) -> bool:
        return all(p.search(sentence) for p in self.evidence)


ACTIONS: tuple[Action, ...] = (
    # Texts say no more than their evidence must contain (generation audit, D4-25).
    Action(
        "ask_error_details",
        en="Could you send us the exact error message you see?",
        de="Könnten Sie uns die genaue Fehlermeldung schicken?",
        evidence=(re.compile(r"\b(?:error\ messages?|fehlermeldung\w*)", _F), ASK),
    ),
    Action(
        "ask_more_details",
        en="Could you share more details about your request?",
        de="Könnten Sie uns weitere Details zu Ihrem Anliegen mitteilen?",
        evidence=(re.compile(
            r"""\b(?:more|additional|further|specific|detailed)\ (?:details|information|info|context)\b
            |(?:weitere|zus[aä]tzliche|genauere|detaillierte|n[aä]here)n?\ (?:details|informationen|angaben)""",
            _F,
        ), ASK),
    ),
    Action(
        "ask_version",
        en="Which software version are you using?",
        de="Welche Softwareversion verwenden Sie?",
        evidence=(re.compile(r"\b(?:versions?|versionen?)\b", _F), ASK),
    ),
    Action(
        "ask_account",
        en="Could you share the account details related to this request?",
        de="Könnten Sie uns die Kontodaten zu diesem Anliegen mitteilen?",
        evidence=(re.compile(
            r"""\b(?:provide|share|send|confirm|include)\b[^.!?]{0,40}\baccount\ (?:number|details|information|id)
            |\b(?:kontonummer|kontodaten|kundennummer|kontoinformationen)\w*[^.!?]{0,40}\b(?:an|mit|bereit|zu)\w*
            |\b(?:geben|teilen|senden|nennen)\ sie\b[^.!?]{0,40}\b(?:kontonummer|kontodaten|kundennummer|kontoinformationen)""",
            _F,
        ),),
    ),
    Action(
        "offer_call",
        en="If it is easier, let us know a convenient time for a phone call.",
        de="Wenn es Ihnen lieber ist, nennen Sie uns gern einen passenden Zeitpunkt für ein Telefonat.",
        evidence=(
            re.compile(r"\b(?:call|phone|telephone|anruf\w*|telefon\w*|r[uü]ckruf\w*|anrufen)\b|<tel_num>", _F),
            re.compile(r"\b(?:times?|availability|available|convenient|suitable|schedule|zeit\w*|termin\w*|erreichbar|verf[uü]gbar)\b", _F),
        ),
    ),
)

ACTIONS_BY_ID = {a.id: a for a in ACTIONS}

GREETING = {"en": "Hello,", "de": "Guten Tag,"}
THANKS = {"en": "Thank you for contacting us.", "de": "vielen Dank für Ihre Nachricht."}
CLOSING = {"en": "Best regards,\nSupport Team", "de": "Mit freundlichen Grüßen\nIhr Support-Team"}

# Fixed lines that carry no claim, per language (normalized for comparison).
BOILERPLATE = {
    lang: {norm(x) for x in (GREETING[lang], THANKS[lang], *CLOSING[lang].split("\n"))}
    for lang in ("en", "de")
}
# Reviewed action texts per language (normalized).
TEMPLATE_TEXTS = {lang: {norm(a.text(lang)) for a in ACTIONS} for lang in ("en", "de")}
