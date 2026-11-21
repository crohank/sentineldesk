"""Orchestration: one `analyze()` used by the API, the console and batch evaluation.

Each stage is a plain callable with the signature from PLAN.md. Teammates replace the
fixture stubs in `components.py` with real implementations; this file should not need
to change when they do.

Decision flow:
  1. mask -> detect language
  2. no usable text -> rule 1 (needs_clarification), skip triage/retrieval/drafting
  3. triage + retrieve -> gate
  4. gate returns status "draft_ready" when no escalation rule fired
  5. draft -> validate (D4's checks plus the contract check below)
  6. abstain or failed validation -> rule 9 (review, DRAFT_FAILED); draft is dropped
"""
from __future__ import annotations

import logging
import json
import time
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Callable, Literal

from .schemas import (
    AnalysisResult,
    Decision,
    Draft,
    GeneratorMode,
    Precedent,
    Reason,
    RetrievalMode,
    StageTimings,
    TicketInput,
    Triage,
    Validation,
)

log = logging.getLogger("sentineldesk")


@dataclass(frozen=True)
class MaskedTicket:
    subject: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}".strip()


@dataclass(frozen=True)
class Components:
    mask: Callable[[TicketInput], MaskedTicket]
    detect_language: Callable[[MaskedTicket], str]
    triage: Callable[[MaskedTicket], Triage] | None
    retrieve: Callable[[MaskedTicket, int], list[Precedent]]
    gate: Callable[[MaskedTicket, str, Triage | None, list[Precedent]], Decision]
    draft: Callable[[MaskedTicket, str, list[Precedent]], Draft]
    validate: Callable[[Draft, str, list[Precedent]], Validation] | None
    release_id: str
    retrieval_mode: RetrievalMode
    generator_mode: GeneratorMode
    artifacts: dict[str, bool]
    triage_with_precedents: Callable[[MaskedTicket, list[Precedent]], Triage] | None = None
    validate_with_ticket: Callable[[MaskedTicket, Draft, str, list[Precedent]], Validation] | None = None


def has_usable_text(body: str) -> bool:
    """Rule 1: the body needs at least one letter. A missing subject is valid."""
    return any(unicodedata.category(ch).startswith("L") for ch in body)


def citation_errors(draft: Draft, precedents: list[Precedent]) -> list[str]:
    """Contract check the console relies on for citation links. Runs on every draft."""
    by_id = {p.ticket_id: p for p in precedents}
    errors = [f"cited {t} was not retrieved" for t in draft.cited_ticket_ids if t not in by_id]
    for i, claim in enumerate(draft.claims):
        if claim.ticket_id not in draft.cited_ticket_ids:
            errors.append(f"claim {i} cites {claim.ticket_id}, which is not in cited_ticket_ids")
        p = by_id.get(claim.ticket_id)
        if p is not None and claim.evidence_span not in p.answer_snippet:
            errors.append(f"claim {i} evidence span not found in {claim.ticket_id}")
    if not draft.abstain and not draft.claims:
        errors.append("draft has no supported claims")
    return errors


class _Clock:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    def run(self, stage: str, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        self.ms[stage] = round((time.perf_counter() - t0) * 1000, 2)
        return out


_default_components: Components | None = None


def default_components() -> Components:
    """Load the configured components once per process. Never per request."""
    global _default_components
    if _default_components is None:
        from .components import load_components

        _default_components = load_components()
    return _default_components


def analyze(
    ticket: TicketInput,
    components: Components | None = None,
    *,
    mode: Literal["live", "batch"] = "live",
    k: int = 5,
) -> AnalysisResult:
    """Run one ticket through the full pipeline.

    `mode="batch"` is for frozen holdout prediction: it disables live-only state
    (burst detection) so evaluation output does not depend on request order.
    """
    c = components or default_components()
    t_start = time.perf_counter()
    clock = _Clock()
    request_id = uuid.uuid4().hex[:12]

    masked = clock.run("masking", c.mask, ticket)
    language = clock.run("language", c.detect_language, masked)

    triage: Triage | None = None
    precedents: list[Precedent] = []
    draft: Draft | None = None
    generator_mode: GeneratorMode = "none"

    if not has_usable_text(masked.body):
        decision = Decision(
            status="needs_clarification",
            destination="requester",
            reasons=[Reason(code="INSUFFICIENT_INPUT", message="Add a short description of the issue.")],
        )
        language = "und"
    else:
        if c.triage_with_precedents is None:
            triage = clock.run("triage", c.triage, masked)
            precedents = clock.run("retrieval", c.retrieve, masked, k)
        else:
            try:
                precedents = clock.run("retrieval", c.retrieve, masked, k)
                triage = clock.run("triage", c.triage_with_precedents, masked, precedents)
            except Exception as exc:
                log.error("managed triage failed request_id=%s error=%s", request_id, type(exc).__name__)

        decision = clock.run("gate", c.gate, masked, language, triage, precedents)
        if triage is None:
            reason = Reason(code="TRIAGE_UNAVAILABLE", message="Managed triage is unavailable.")
            if decision.status == "draft_ready":
                decision = Decision(status="review", destination="human_review", reasons=[reason])
            else:
                decision.reasons.append(reason)

        if decision.status == "draft_ready":
            generator_mode = c.generator_mode
            candidate: Draft | None = None
            try:
                candidate = clock.run("generation", c.draft, masked, language, precedents)
                if c.validate_with_ticket is not None:
                    validation = clock.run("validation", c.validate_with_ticket, masked, candidate, language, precedents)
                else:
                    validation = clock.run("validation", c.validate, candidate, language, precedents)
                errors = list(validation.errors) + citation_errors(candidate, precedents)
                failed = candidate.abstain or not validation.valid or bool(errors)
            except Exception as exc:  # rule 9: a generator or validator failure is a review, not a 500
                # Type name only: exception messages may quote ticket text.
                log.error("draft stage failed request_id=%s error=%s", request_id, type(exc).__name__)
                errors = [f"{'generation' if candidate is None else 'validation'} error ({type(exc).__name__})"]
                failed = True
            if failed:
                detail = "; ".join(errors) if errors else "generator abstained"
                decision = Decision(
                    status="review",
                    destination="human_review",
                    reasons=[Reason(code="DRAFT_FAILED", message=f"Draft withheld: {detail}.")],
                )
            else:
                draft = candidate

    result = AnalysisResult(
        request_id=request_id,
        language=language,
        triage=triage,
        decision=decision,
        precedents=precedents,
        draft=draft,
        stage_ms=StageTimings(**clock.ms),
        total_ms=round((time.perf_counter() - t_start) * 1000, 2),
        release_id=c.release_id,
        retrieval_mode=c.retrieval_mode,
        generator_mode=generator_mode,
    )
    _log_result(result)
    return result


def _log_result(r: AnalysisResult) -> None:
    """Structured log line. Never includes ticket text, answers or prompts."""
    log.info(
        json.dumps(
            {
                "request_id": r.request_id,
                "release_id": r.release_id,
                "language": r.language,
                "predicted": None
                if r.triage is None
                else {"type": r.triage.type.label, "queue": r.triage.queue.label, "priority": r.triage.priority.label},
                "status": r.decision.status,
                "reason_codes": [x.code for x in r.decision.reasons],
                "retrieval_mode": r.retrieval_mode,
                "generator_mode": r.generator_mode,
                "stage_ms": r.stage_ms.model_dump(exclude_none=True),
                "total_ms": r.total_ms,
            }
        )
    )
