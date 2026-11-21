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
    Action(
        "ask_error_details",
        en="Could you send us the exact error message or code you see, and tell us when it first appeared?",
        de="Könnten Sie uns die genaue Fehlermeldung oder den Fehlercode schicken und mitteilen, seit wann sie auftritt?",
        evidence=(re.compile(
            r"""\b(?:error\ (?:messages?|codes?|details|logs?)|logs?\b|log\ files?|screenshots?|fehlermeldung\w*
            |fehlercode\w*|protokoll\w*|logdatei\w*|screenshot\w*|bildschirmfoto\w*)""",
            _F,
        ), ASK),
    ),
    Action(
        "ask_more_details",
        en="Could you share a few more details about the issue, such as what you were doing when it occurred?",
        de="Könnten Sie uns noch etwas mehr zum Problem mitteilen, zum Beispiel, was Sie gerade getan haben, als es auftrat?",
        evidence=(re.compile(
            r"""\b(?:more|additional|further|specific|detailed)\ (?:details|information|info|context)\b
            |(?:weitere|zus[aä]tzliche|genauere|detaillierte|n[aä]here)n?\ (?:details|informationen|angaben)""",
            _F,
        ), ASK),
    ),
    Action(
        "ask_environment",
        en="Which software version, device and operating system are you using?",
        de="Welche Softwareversion, welches Gerät und welches Betriebssystem verwenden Sie?",
        evidence=(re.compile(
            r"""\b(?:versions?|devices?|operating\ systems?|browsers?|ger[aä]t\w*|betriebssystem\w*)\b""",
            _F,
        ), ASK),
    ),
    Action(
        "ask_account",
        en="Please let us know which account this concerns so the team can look into it.",
        de="Bitte teilen Sie uns mit, welches Konto betroffen ist, damit das Team die Angelegenheit prüfen kann.",
        evidence=(re.compile(
            r"""account\ (?:number|details|information|id|name)|<acc_num>|customer\ (?:number|id)|kontonummer
            |kontodaten|kundennummer|kontoinformationen|kundenkonto""",
            _F,
        ), ASK),
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
LEAD = {"en": "To move this forward:", "de": "Damit wir weiterhelfen können:"}
CLOSING = {"en": "Best regards,\nSupport Team", "de": "Mit freundlichen Grüßen\nIhr Support-Team"}

# Fixed lines that carry no claim, per language (normalized for comparison).
BOILERPLATE = {
    lang: {norm(x) for x in (GREETING[lang], THANKS[lang], LEAD[lang], *CLOSING[lang].split("\n"))}
    for lang in ("en", "de")
}
# Reviewed action texts per language (normalized).
TEMPLATE_TEXTS = {lang: {norm(a.text(lang)) for a in ACTIONS} for lang in ("en", "de")}
