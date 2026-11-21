"""English and German patterns for gate rules 2, 3, 5 and 8. Case-insensitive.

Each detector returns the matched words (for the operator message) or nothing.
Trigger rates on the KB validation partition are in reports/d4/rule_rates.md.
"""
from __future__ import annotations

import re

from .text import sentences

_F = re.IGNORECASE | re.VERBOSE

# ---------------------------------------------------------------- rule 2: high-impact outage
# Fires only when all three groups match: an ongoing outage, a broad scope and a critical service.

OUTAGE = re.compile(
    r"""
    \b(?:outages?|offline|unavailable|unreachable|inaccessible|went\ down
    |(?:is|are|was|were|has\ been|have\ been|been)\ down|down\ (?:since|for|across|at|in)
    |not\ (?:working|available|reachable|accessible|responding|loading)|stopped\ working
    |(?:can\ ?not|can't|cannot|unable\ to|could\ not|couldn't)\ (?:connect|access|log\ ?in|sign\ ?in|reach|use|open)
    |no\ (?:access|connection|connectivity)|interrupt(?:ed|ion)|disrupt(?:ed|ion)|crash(?:ed|es|ing)?
    |ausfall\w*|ausf[aä]ll\w*|ausgefallen|nicht\ (?:mehr\ )?(?:erreichbar|verf[uü]gbar|m[oö]glich|nutzbar|zug[aä]nglich)
    |nicht\ (?:mehr\ )?(?:\w+\ ){0,3}?(?:verbinden|anmelden|einloggen|zugreifen)
    |keine\ verbindung|kein(?:en)?\ (?:zugriff|zugang)|funktionier\w*\ nicht|gest[oö]rt|st[oö]rung\w*
    |abgest[uü]rzt|unterbrochen)\b
    """,
    _F,
)

SCOPE = re.compile(
    r"""
    \b(?:everyone|everybody|nobody|no\ one|company[-\ ]wide|organi[sz]ation[-\ ]wide|enterprise[-\ ]wide
    |(?:all|every|entire|whole)\ (?:of\ )?(?:our\ |the\ |my\ )?(?:users?|staff|employees?|customers?|clients?|sites?
      |locations?|offices?|branches|teams?|departments?|clinics?|hospitals?|stores?|regions?|company
      |organi[sz]ation|network|infrastructure|systems?|servers?)
    |across\ (?:all|multiple|several)\ \w+
    |multiple\ (?:sites|locations|offices|branches|teams|departments|regions|users|customers)
    |alle[nrs]?\ (?:\w+\ )?(?:mitarbeitende\w*|mitarbeiter\w*|nutzer\w*|benutzer\w*|kunden|standort\w*|filialen
      |abteilungen|systeme|server|kliniken|praxen)
    |an\ allen\ standorten|s[aä]mtliche\w*\ \w+|gesamte[nrs]?\ \w+
    |ganze[nrs]?\ (?:unternehmen|netzwerk|firma|standort|abteilung|team|klinik|krankenhaus)
    |niemand|unternehmensweit|standort[uü]bergreifend|fl[aä]chendeckend)\b
    """,
    _F,
)

CRITICAL = re.compile(
    r"""
    \b(?:vpn|network|internet|wi-?fi|e-?mail|mail\ server|servers?|databases?|production|payments?
    |payment\ (?:system|gateway)|checkout|login|log-?in|sign-?in|authentication|sso|single\ sign-?on
    |active\ directory|emr|ehr|(?:electronic\ )?(?:health|medical|patient)\ records?|telemedicine|telehealth
    |erp|crm|data\ ?cent(?:er|re)|cloud|platform|website|portal|phone\ system|telephony
    |netzwerk\w*|wlan|mailserver|datenbank\w*|produktion\w*|zahlung\w*|kasse\w*|anmeldung|authentifizierung
    |patientenakte\w*|telemedizin|rechenzentrum|plattform|webseite|telefonanlage)\b
    """,
    _F,
)

# ---------------------------------------------------------------- rule 3: instruction override

OVERRIDE = re.compile(
    r"""
    \b(?:ignore|disregard|forget|override|bypass)\b[^.!?\n]{0,40}?
      \b(?:previous|prior|above|earlier|all|your|the|any|system|safety)\b[^.!?\n]{0,25}?
      \b(?:instructions?|prompts?|rules?|guidelines?|polic(?:y|ies)|guardrails?|restrictions?)\b
    |\byou\ are\ now\b|\bsystem\ prompt\b|\bdeveloper\ mode\b|\bjailbreak\w*
    |\bas\ (?:the|your|an?)\ (?:admin(?:istrator)?|ceo|cto|cio|manager|supervisor|director|owner)\b[^.!?\n]{0,40}?
      \b(?:authori[sz]e|approve|order|instruct|command)\w*
    |\b(?:ignorier\w*|missacht\w*|vergiss|vergessen\ sie|[uü]bergeh\w*)\b[^.!?\n]{0,40}?
      \b(?:bisherig\w*|vorherig\w*|obig\w*|alle\w*|deine\w*|ihre\w*|system\w*)\b[^.!?\n]{0,25}?
      \b(?:anweisung\w*|regel\w*|vorgabe\w*|richtlinie\w*|prompt\w*|instruktion\w*)
    |\bdu\ bist\ (?:jetzt|nun|ab\ sofort)\b|\bsystem-?prompt\w*|\bentwicklermodus\b
    |\bals\ (?:ihr|dein|der|ihre)\ (?:administrator|admin|gesch[aä]ftsf[uü]hrer\w*|vorgesetzte\w*|chef\w*|manager\w*)\b
      [^.!?\n]{0,40}?\b(?:genehmig|autorisier|ordne|weise|befehl)\w*
    """,
    _F,
)

# ---------------------------------------------------------------- rule 5: authorization required
# A commitment term plus a demand cue in the same sentence, or a strong phrase on its own.
# "Where can I view my invoice?" has no commitment term, so it never fires.

COMMITMENT = re.compile(
    r"""
    \b(?:refund\w*|reimburs\w*|money\ back|compensat\w*|(?:sla|service)[-\ ]credits?|credit\ note|chargebacks?
    |discounts?|price\ (?:guarantee|lock|match|freeze)|binding\ (?:quote|offer|price)|guarantee\w*|gdpr
    |data\ subject\ (?:access\ )?requests?|right\ to\ (?:be\ forgotten|erasure)
    |(?:delete|erase|wipe|purge)\ (?:all|every|my|our|the\ entire)\b[^.!?\n]{0,30}?\b(?:data|records|accounts?|users|information|files|history)
    |erstatt\w*|r[uü]ckerstatt\w*|geld\ zur[uü]ck|entsch[aä]dig\w*|gutschrift\w*|gutschreib\w*|rabatt\w*
    |preisgarantie|festpreis\w*|verbindlich\w*\ (?:angebot|preis)\w*|garanti\w*|dsgvo|datenschutz-?grundverordnung
    |auskunftsersuchen|recht\ auf\ (?:l[oö]schung|vergessenwerden)
    |(?:alle|s[aä]mtliche|meine|unsere)\ (?:\w+\ )?daten\ (?:l[oö]schen|vernichten)
    |l[oö]schen\ sie\ (?:alle|s[aä]mtliche|meine|unsere))\b
    """,
    _F,
)

DEMAND = re.compile(
    r"""
    \b(?:i|we)\ (?:want|demand|require|expect|insist|request|need|would\ like|deserve|am\ owed|are\ owed)\b
    |\bentitled\b|\b(?:confirm|guarantee|promise|ensure|commit)\b|\bgive\ (?:me|us)\b|\bissue\ (?:me|us|a|an|the)\b
    |\b(?:please|kindly)\ (?:refund|reimburse|compensate|credit|delete|erase|confirm|guarantee|send|process|issue|apply)\b
    |\bcan\ (?:i|we)\ (?:get|have|receive|claim)\b|\bowe[ds]?\b|\bmust\b|\bimmediate(?:ly)?\b|\bdemand\w*
    |\b(?:ich|wir)\ (?:verlange\w*|fordere\w*|erwarte\w*|m[oö]chte\w*|will|wollen|brauche\w*|ben[oö]tige\w*|bestehe\w*)\b
    |\banspruch\w*|\bbest[aä]tigen\ sie\b|\bgarantieren\ sie\b
    |\bbitte\ (?:erstatten|entsch[aä]digen|l[oö]schen|best[aä]tigen|gutschreiben|veranlassen)\b
    |\bsteh(?:t|en)\ (?:mir|uns)\b[^.!?\n]{0,20}\bzu\b|\bschulde\w*|\bsofort\w*|\bunverz[uü]glich\w*|\bforder\w*|\bverlang\w*
    """,
    _F,
)

STRONG = re.compile(
    r"""
    \brefund\ (?:me|us)\b|\b(?:full|partial|complete)\ refund\b|\bgive\ (?:me|us)\ (?:a\ |my\ |our\ )?(?:money|refund)\b
    |\bdelete\ all\b|\b(?:gdpr|data\ subject)\ (?:request|deletion|erasure|access)\b
    |\berstatten\ sie\ (?:mir|uns)\b|\b(?:volle|vollst[aä]ndige|komplette)n?\ (?:erstattung|r[uü]ckerstattung)\b
    |\balle\ (?:meine\ |unsere\ )?daten\ l[oö]schen\b|\bdsgvo-?(?:anfrage|antrag|auskunft|l[oö]schung)\b
    """,
    _F,
)

# ---------------------------------------------------------------- rule 8: integration requests

INTEGRATION = re.compile(
    r"""
    \b(?:integrat\w*|compatib\w*|interoperab\w*|connectors?|plug-?ins?|add-?ons?
    |integration\w*|integrier\w*|kompatib\w*|anbindung\w*|anbinden|schnittstelle\w*)\b
    """,
    _F,
)

REQUEST_CUE = re.compile(
    r"""
    \?|\b(?:can|could|would|does|do\ you|is\ it\ possible|how\ (?:to|do|can|should)|please|need|want|looking\ for
    |seeking|guidance|help|assist\w*|k[oö]nn\w*|ist\ es\ m[oö]glich|wie|bitte|ben[oö]tig\w*|brauche\w*|such\w*
    |unterst[uü]tz\w*|hilfe|anleitung)\b
    """,
    _F,
)


def has_usable_text(body: str | None) -> bool:
    """Rule 1, same test as the pipeline: the body needs at least one letter."""
    return any(ch.isalpha() for ch in body or "")


def outage_impact(text: str) -> dict[str, str] | None:
    outage, scope, service = OUTAGE.search(text), SCOPE.search(text), CRITICAL.search(text)
    if outage and scope and service:
        return {"outage": outage.group(0), "scope": scope.group(0), "service": service.group(0)}
    return None


def instruction_override(text: str) -> str | None:
    m = OVERRIDE.search(text)
    return " ".join(m.group(0).split()) if m else None


def authorization_terms(text: str) -> list[str]:
    found: list[str] = []
    for sentence in sentences(text):
        hits = [m.group(0) for m in STRONG.finditer(sentence)]
        if not hits and DEMAND.search(sentence):
            hits = [m.group(0) for m in COMMITMENT.finditer(sentence)]
        for h in hits:
            term = " ".join(h.lower().split())
            if term not in found:
                found.append(term)
    return found


def integration_request(text: str) -> str | None:
    for sentence in sentences(text):
        m = INTEGRATION.search(sentence)
        if m and REQUEST_CUE.search(sentence):
            return m.group(0)
    return None
