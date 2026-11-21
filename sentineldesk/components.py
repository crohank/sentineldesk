"""Component wiring. Fixture stubs today; real D2/D3/D4 components replace them.

To plug in a real component, write a function with the same signature and swap it into
`load_components()`. Set `retrieval_mode` / `generator_mode` to the real mode so the
console and the logs say what actually ran.

Fixture stubs return the synthetic data in `fixtures/routine.json` for every ticket that
has usable text. Every fixture string starts with "[FIXTURE]" and the console shows a
banner, so nobody mistakes this for a real prediction.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources

from langdetect import DetectorFactory, LangDetectException, detect

from .pipeline import Components, MaskedTicket
from .schemas import AnalysisResult, Decision, Draft, Precedent, TicketInput, Triage, Validation

DetectorFactory.seed = 0  # langdetect is random without a seed

FIXTURE_NAMES = ("routine", "on_call", "insufficient")


@lru_cache(maxsize=None)
def load_fixture(name: str) -> AnalysisResult:
    if name not in FIXTURE_NAMES:
        raise KeyError(name)
    raw = resources.files("sentineldesk.fixtures").joinpath(f"{name}.json").read_text(encoding="utf-8")
    return AnalysisResult.model_validate(json.loads(raw))


# ------------------------------------------------------------------ stubs


def mask_passthrough(ticket: TicketInput) -> MaskedTicket:
    # TODO(D4): replace with PII masking (emails, phones, IPs, account IDs).
    return MaskedTicket(subject=ticket.subject or "", body=ticket.body)


def detect_language_langdetect(ticket: MaskedTicket) -> str:
    # TODO(D3): replace with the release detector configured for out-of-scope languages.
    try:
        return detect(ticket.text)
    except LangDetectException:
        return "und"


def triage_fixture(ticket: MaskedTicket) -> Triage:
    return load_fixture("routine").triage  # type: ignore[return-value]


def retrieve_fixture(ticket: MaskedTicket, k: int = 5) -> list[Precedent]:
    return load_fixture("routine").precedents[:k]


def gate_fixture(ticket: MaskedTicket, language: str, triage: Triage, precedents: list[Precedent]) -> Decision:
    return load_fixture("routine").decision


def draft_fixture(ticket: MaskedTicket, language: str, precedents: list[Precedent]) -> Draft:
    return load_fixture("routine").draft  # type: ignore[return-value]


def validate_passthrough(draft: Draft, language: str, precedents: list[Precedent]) -> Validation:
    # TODO(D4): claim, numeric, qualifier and language checks. The pipeline already
    # runs the citation contract check on every draft.
    return Validation(valid=True)


# ------------------------------------------------------------------ wiring


def fixture_components() -> Components:
    detect_language_langdetect(MaskedTicket("", "warm up the language profiles"))  # first call loads ~170 ms of profiles
    return Components(
        mask=mask_passthrough,
        detect_language=detect_language_langdetect,
        triage=triage_fixture,
        retrieve=retrieve_fixture,
        gate=gate_fixture,
        draft=draft_fixture,
        validate=validate_passthrough,
        release_id=os.environ.get("SENTINELDESK_RELEASE_ID", "dev-fixture"),
        retrieval_mode="fixture",
        generator_mode="fixture",
        artifacts={"triage_model": False, "index": False, "evidence_store": False},
    )


def load_components() -> Components:
    mode = os.environ.get("SENTINELDESK_MODE", "fixture")
    if mode == "fixture":
        return fixture_components()
    if mode == "managed":
        return managed_components()
    raise ValueError(f"unknown SENTINELDESK_MODE={mode!r}; expected fixture or managed")


def managed_components() -> Components:
    from pathlib import Path

    from .agent_retrieval import AgentRetrieval
    from .gemini_triage import GeminiTriage
    from .safety.pii import mask_ticket
    from .schemas import Reason

    store = AgentRetrieval()
    data_dir = Path(os.environ.get("SENTINELDESK_DATA_DIR", "data/processed"))
    model = os.environ.get("SENTINELDESK_TRIAGE_MODEL", "gemini-3.8-flash")
    classifier = GeminiTriage(store, data_dir, model)

    def review(ticket, language, triage, precedents):
        return Decision(
            status="review",
            destination="human_review",
            reasons=[Reason(code="OPERATOR_REVIEW", message="Review the classification and precedents.")],
        )

    return Components(
        mask=mask_ticket,
        detect_language=detect_language_langdetect,
        triage=None,
        retrieve=store.retrieve,
        gate=review,
        draft=None,
        validate=None,
        release_id=os.environ.get("SENTINELDESK_RELEASE_ID", "managed-triage"),
        retrieval_mode="agent_retrieval",
        generator_mode="none",
        artifacts={"training_examples": True, "drafting": False},
        triage_with_precedents=classifier.predict,
    )
