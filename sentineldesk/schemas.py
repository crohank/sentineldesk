"""Shared contract for every SentinelDesk component.

Field names follow PLAN.md "Interfaces to freeze by 10:40" exactly. D2 (triage),
D3 (retrieval), D4 (gate, drafting) and D5 (API, console) all import from here.

Label values (type, queue, priority) stay plain strings because the queue names are
learned from the KB, not hardcoded.

Reason codes, statuses and destinations are PROVISIONAL until D4 confirms them.
They map one-to-one onto the ordered escalation rules 1-9 in PLAN.md.

Regenerate the committed JSON Schema after any change:
    uv run export_schemas
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- request


class TicketInput(_Model):
    """The only body `POST /api/triage` accepts.

    `body` has no minimum length on purpose: an empty body is a valid request that
    gate rule 1 answers with `needs_clarification` (HTTP 200), not a 422.
    """

    subject: str | None = None
    body: str


# ---------------------------------------------------------------- triage (D2)


class Prediction(_Model):
    label: str
    confidence: float = Field(ge=0.0, le=1.0, description="Calibrated probability of `label`.")
    probabilities: dict[str, float] = Field(description="Calibrated probability per class.")


class Triage(_Model):
    type: Prediction
    queue: Prediction
    priority: Prediction


# ---------------------------------------------------------------- retrieval (D3)


class Precedent(_Model):
    ticket_id: str = Field(pattern=r"^TKT-\d{5}$")
    score: float
    subject: str
    answer_snippet: str = Field(description="Redacted excerpt of the historical answer.")
    answer_language: str = Field(description="Detected from the answer text, not the metadata column.")


# ---------------------------------------------------------------- gate (D4)

# Rule order from PLAN.md. Rule 2 has two triggers, so it has two codes.
ReasonCode = Literal[
    "INSUFFICIENT_INPUT",  # 1
    "HIGH_IMPACT_OUTAGE",  # 2 impact rule: ongoing outage + broad impact + critical service
    "PREDICTED_INCIDENT_HIGH",  # 2 classifier: type == Incident and priority == high
    "INSTRUCTION_OVERRIDE",  # 3
    "UNSUPPORTED_LANGUAGE",  # 4
    "AUTHORIZATION_REQUIRED",  # 5 refund, SLA credit, GDPR demand, delete all, ...
    "OUTAGE_RISK",  # 6 Incident and high probabilities both above risk thresholds
    "LOW_CONFIDENCE",  # 7
    "WEAK_EVIDENCE",  # 8
    "DRAFT_FAILED",  # 9 abstain, timeout, or failed validation
]

DecisionStatus = Literal[
    "needs_clarification",  # rule 1
    "escalate",  # rules 2 and 6: on-call
    "review",  # rules 3, 4, 5, 7, 8, 9
    "draft_ready",  # no rule fired and the draft validated
]

Destination = Literal[
    "requester",  # rule 1: ask the customer for more detail
    "on_call",
    "human_review",
    "language_review",
    "specialist_review",
    "operator",  # draft_ready: operator reviews the draft before anything is sent
]


class Reason(_Model):
    code: ReasonCode
    message: str = Field(description="Operator-facing text.")


class Decision(_Model):
    status: DecisionStatus
    destination: Destination
    reasons: list[Reason] = Field(default_factory=list)


# ---------------------------------------------------------------- drafting (D4)


class Claim(_Model):
    text: str
    ticket_id: str = Field(pattern=r"^TKT-\d{5}$")
    evidence_span: str = Field(description="Verbatim substring of the cited precedent's answer.")


class Draft(_Model):
    """Operator draft. Mark each citation inline in `reply_text` as `[TKT-12345]`.

    The console turns those markers into links to the matching evidence card. Every
    marked ID must appear in `cited_ticket_ids`, and every cited ID must be a
    retrieved precedent.
    """

    reply_text: str
    cited_ticket_ids: list[str]
    claims: list[Claim]
    abstain: bool


class Validation(_Model):
    valid: bool
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- response (D5)

RetrievalMode = Literal["fixture", "sparse", "dense"]
GeneratorMode = Literal["fixture", "offline_templates", "gemma", "none"]


class StageTimings(_Model):
    """Milliseconds per stage. A stage that did not run is null."""

    masking: float | None = None
    language: float | None = None
    triage: float | None = None
    retrieval: float | None = None
    gate: float | None = None
    generation: float | None = None
    validation: float | None = None


class BurstStatus(_Model):
    active: bool
    service: str | None = None
    ticket_count: int = 0
    window_minutes: int = 10


class AnalysisResult(_Model):
    request_id: str
    language: str = Field(description="Detected from ticket text: 'en', 'de', other ISO code, or 'und'.")
    triage: Triage | None = Field(description="Null when gate rule 1 skips prediction.")
    decision: Decision
    precedents: list[Precedent]
    draft: Draft | None = Field(description="Only a draft that passed validation is ever returned.")
    stage_ms: StageTimings
    total_ms: float
    release_id: str
    retrieval_mode: RetrievalMode
    generator_mode: GeneratorMode
    burst: BurstStatus | None = None


class Health(_Model):
    status: Literal["ok", "degraded"]
    release_id: str
    retrieval_mode: RetrievalMode
    generator_mode: GeneratorMode
    artifacts: dict[str, bool] = Field(description="Readiness per artifact name.")


# ---------------------------------------------------------------- export

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "schemas" / "contract.schema.json"

EXPORTED = {
    "TicketInput": TicketInput,
    "AnalysisResult": AnalysisResult,
    "Health": Health,
    "Validation": Validation,
}


def contract_json_schema() -> dict:
    return {name: model.model_json_schema() for name, model in EXPORTED.items()}


def export_main() -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(contract_json_schema(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {SCHEMA_PATH}")
