"""Gate rules 1-8 (PLAN.md, "Ordered escalation rules").

Returns the shared contract's Decision. When several rules fire, on-call wins over
review and review wins over draft_ready; the reasons of every rule that fired are kept
so the console can show all of them. Rule 1 is repeated here (same test as the
pipeline) so the gate is safe to call on its own. Rule 9 runs after drafting.
"""
from __future__ import annotations

from . import contracts as c
from .config import DEFAULT_CONFIG, SafetyConfig
from .patterns import (
    authorization_terms,
    has_usable_text,
    instruction_override,
    integration_request,
    outage_impact,
)

SUPPORTED_LANGUAGES = ("en", "de")
HEADS = ("type", "queue", "priority")
ONCALL_CODES = {c.HIGH_IMPACT_OUTAGE, c.PREDICTED_INCIDENT_HIGH, c.OUTAGE_RISK}


def normalize_language(language: str | None) -> str:
    lang = (language or "").strip().lower().replace("_", "-").split("-")[0]
    return lang or "und"


def _label(pred) -> str:
    return str(getattr(pred, "label", "") or "")


def _conf(pred) -> float:
    return float(getattr(pred, "confidence", 0.0) or 0.0)


def _prob(pred, label: str) -> float:
    probs = getattr(pred, "probabilities", None) or {}
    if label in probs:
        return float(probs[label])
    return _conf(pred) if _label(pred) == label else 0.0


def _answer(p) -> str:
    return getattr(p, "answer_snippet", "") or ""


def gate(masked_ticket, language, triage, precedents, *, config: SafetyConfig = DEFAULT_CONFIG) -> c.Decision:
    subject = getattr(masked_ticket, "subject", "") or ""
    body = getattr(masked_ticket, "body", "") or ""

    # Rule 1: no usable text. Skip everything else.
    if not has_usable_text(body):
        return c.Decision(
            status=c.NEEDS_CLARIFICATION,
            destination=c.REQUESTER,
            reasons=[c.Reason(code=c.INSUFFICIENT_INPUT, message="Add a short description of the issue.")],
        )

    text = f"{subject}\n{body}"
    reasons: list[c.Reason] = []
    review_destination: str | None = None

    # Rule 2: high-impact outage, by explicit wording or by the classifier.
    hit = outage_impact(text)
    if hit:
        reasons.append(c.Reason(
            code=c.HIGH_IMPACT_OUTAGE,
            message=f"Reports a widespread outage of a critical service: \"{hit['outage']}\", "
                    f"\"{hit['scope']}\", \"{hit['service']}\".",
        ))
    incident_high = False
    if triage is not None and _label(triage.type) == "Incident" and _label(triage.priority) == "high":
        incident_high = True
        reasons.append(c.Reason(
            code=c.PREDICTED_INCIDENT_HIGH,
            message=f"Classifier predicts Incident ({_conf(triage.type):.0%}) with high priority "
                    f"({_conf(triage.priority):.0%}).",
        ))

    # Rule 3: instruction override or fabricated authority.
    override = instruction_override(text)
    if override:
        reasons.append(c.Reason(
            code=c.INSTRUCTION_OVERRIDE,
            message=f"Ticket requests an override of drafting instructions (\"{override}\"). No draft is written.",
        ))
        review_destination = review_destination or c.HUMAN_REVIEW

    # Rule 4: unsupported or undetermined language.
    lang = normalize_language(language)
    if lang not in SUPPORTED_LANGUAGES:
        shown = "undetermined" if lang == "und" else lang
        reasons.append(c.Reason(
            code=c.UNSUPPORTED_LANGUAGE,
            message=f"Detected language: {shown}. Drafts are written only in English or German.",
        ))
        review_destination = review_destination or c.LANGUAGE_REVIEW

    # Rule 5: request needs authorization or could be destructive.
    terms = authorization_terms(text)
    if terms:
        reasons.append(c.Reason(
            code=c.AUTHORIZATION_REQUIRED,
            message=f"Asks for a commitment that needs authorization ({', '.join(terms)}). "
                    "A past resolution for another customer cannot approve it.",
        ))
        review_destination = review_destination or c.SPECIALIST_REVIEW

    if triage is not None:
        # Rule 6: Incident and high both likely, even as runner-up labels.
        p_incident, p_high = _prob(triage.type, "Incident"), _prob(triage.priority, "high")
        min_incident, min_high = config.outage_pair
        if not incident_high and p_incident >= min_incident and p_high >= min_high:
            reasons.append(c.Reason(
                code=c.OUTAGE_RISK,
                message=f"Potential high-priority Incident: P(Incident) {p_incident:.2f}, P(high) {p_high:.2f}, "
                        f"although the predicted labels are {_label(triage.type)} / {_label(triage.priority)}.",
            ))
        # Rule 7: any head below its confidence threshold.
        for head in HEADS:
            pred = getattr(triage, head)
            threshold = config.head_thresholds[head]
            if _conf(pred) < threshold:
                reasons.append(c.Reason(
                    code=c.LOW_CONFIDENCE,
                    message=f"Uncertain {head}: {_label(pred)} at {_conf(pred):.2f}, below {threshold:.2f}.",
                ))
    else:
        reasons.append(c.Reason(code=c.LOW_CONFIDENCE, message="No triage prediction is available."))

    # Rule 8: weak or missing evidence.
    precedents = list(precedents or [])
    floor = config.evidence_threshold()
    best = max((float(getattr(p, "score", 0.0)) for p in precedents), default=None)
    if best is None or best < floor:
        shown = "none retrieved" if best is None else f"best similarity {best:.2f}"
        reasons.append(c.Reason(
            code=c.WEAK_EVIDENCE,
            message=f"No sufficiently similar past ticket ({shown}; needs {floor:.2f}).",
        ))
    elif not any(_answer(p).strip() for p in precedents if float(getattr(p, "score", 0.0)) >= floor):
        reasons.append(c.Reason(code=c.WEAK_EVIDENCE, message="Similar past tickets have no usable answer."))
    integration = integration_request(text)
    if integration:
        reasons.append(c.Reason(
            code=c.WEAK_EVIDENCE,
            message=f"Asks about an integration or compatibility (\"{integration}\"). Past answers can't confirm "
                    "it for this customer.",
        ))

    if any(r.code in ONCALL_CODES for r in reasons):
        return c.Decision(status=c.ESCALATE, destination=c.ON_CALL, reasons=reasons)
    if reasons:
        return c.Decision(status=c.REVIEW, destination=review_destination or c.HUMAN_REVIEW, reasons=reasons)
    return c.Decision(status=c.DRAFT_READY, destination=c.OPERATOR, reasons=[])
