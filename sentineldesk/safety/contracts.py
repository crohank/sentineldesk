"""Types at D4's boundaries, mirroring the shared contract.

The team contract is `sentineldesk/schemas.py` (D5), plus `MaskedTicket` in
`sentineldesk/pipeline.py`. These dataclasses copy their field names and allowed values
exactly, so every object D4 builds is accepted by the shared Pydantic models as is.

Once the shared contract is on this branch, replace the class definitions below with

    from sentineldesk.schemas import (Claim, Decision, Draft, Precedent, Prediction,
                                      Reason, TicketInput, Triage, Validation)
    from sentineldesk.pipeline import MaskedTicket

and keep the constants. Nothing else in `sentineldesk.safety` changes: every function
reads its inputs by attribute and builds its outputs with keyword arguments.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Decision statuses (shared contract: DecisionStatus)
NEEDS_CLARIFICATION = "needs_clarification"
ESCALATE = "escalate"
REVIEW = "review"
DRAFT_READY = "draft_ready"
STATUSES = (NEEDS_CLARIFICATION, ESCALATE, REVIEW, DRAFT_READY)

# Destinations (shared contract: Destination)
REQUESTER = "requester"
ON_CALL = "on_call"
HUMAN_REVIEW = "human_review"
LANGUAGE_REVIEW = "language_review"
SPECIALIST_REVIEW = "specialist_review"
OPERATOR = "operator"
DESTINATIONS = (REQUESTER, ON_CALL, HUMAN_REVIEW, LANGUAGE_REVIEW, SPECIALIST_REVIEW, OPERATOR)

# Reason codes (shared contract: ReasonCode), in rule order
INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT"  # 1
HIGH_IMPACT_OUTAGE = "HIGH_IMPACT_OUTAGE"  # 2, impact patterns
PREDICTED_INCIDENT_HIGH = "PREDICTED_INCIDENT_HIGH"  # 2, classifier
INSTRUCTION_OVERRIDE = "INSTRUCTION_OVERRIDE"  # 3
UNSUPPORTED_LANGUAGE = "UNSUPPORTED_LANGUAGE"  # 4
AUTHORIZATION_REQUIRED = "AUTHORIZATION_REQUIRED"  # 5
OUTAGE_RISK = "OUTAGE_RISK"  # 6
LOW_CONFIDENCE = "LOW_CONFIDENCE"  # 7
WEAK_EVIDENCE = "WEAK_EVIDENCE"  # 8
DRAFT_FAILED = "DRAFT_FAILED"  # 9
REASON_CODES = (
    INSUFFICIENT_INPUT,
    HIGH_IMPACT_OUTAGE,
    PREDICTED_INCIDENT_HIGH,
    INSTRUCTION_OVERRIDE,
    UNSUPPORTED_LANGUAGE,
    AUTHORIZATION_REQUIRED,
    OUTAGE_RISK,
    LOW_CONFIDENCE,
    WEAK_EVIDENCE,
    DRAFT_FAILED,
)

# Generator modes (shared contract: GeneratorMode)
OFFLINE_TEMPLATES = "offline_templates"
GEMMA = "gemma"

# The scorer (docs/prediction_format.md, D1) names the statuses differently.
SCORER_STATUS = {
    DRAFT_READY: "auto_draft",
    ESCALATE: "escalate_oncall",
    REVIEW: "human_review",
    NEEDS_CLARIFICATION: "needs_clarification",
}

# Badge text for the console (brief: "Auto-Draft Ready" vs "Escalate to On-Call Engineer").
STATUS_BADGES = {
    DRAFT_READY: "Auto-Draft Ready",
    ESCALATE: "Escalate to On-Call Engineer",
    REVIEW: "Human Review",
    NEEDS_CLARIFICATION: "Needs Clarification",
}


@dataclass
class TicketInput:
    body: str
    subject: str | None = None


@dataclass(frozen=True)
class MaskedTicket:
    subject: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}".strip()


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class Triage:
    type: Prediction
    queue: Prediction
    priority: Prediction


@dataclass
class Precedent:
    ticket_id: str
    score: float
    subject: str
    answer_snippet: str
    answer_language: str


@dataclass
class Reason:
    code: str
    message: str


@dataclass
class Decision:
    status: str
    destination: str
    reasons: list[Reason] = field(default_factory=list)


@dataclass
class Claim:
    text: str
    ticket_id: str
    evidence_span: str


@dataclass
class Draft:
    reply_text: str
    cited_ticket_ids: list[str]
    claims: list[Claim]
    abstain: bool


@dataclass
class Validation:
    valid: bool
    errors: list[str] = field(default_factory=list)
