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
    from dataclasses import asdict, replace
    from pathlib import Path

    from .agent_retrieval import AgentRetrieval
    from .gemini_triage import GeminiTriage
    from .safety.config import DEFAULT_CONFIG
    from .safety.drafting import draft_template
    from .safety.gate import gate
    from .safety.pii import mask_pii, mask_ticket
    from .safety.validation import validate

    store = AgentRetrieval()
    data_dir = Path(os.environ.get("SENTINELDESK_DATA_DIR", "data/processed"))
    model = os.environ.get("SENTINELDESK_TRIAGE_MODEL", "gemini-3.8-flash")
    gemini = GeminiTriage(store, data_dir, model)
    from .hybrid import HybridTriage
    from .search_snapshot import SearchSnapshot
    bundle = Path(os.environ.get("SENTINELDESK_MODEL_DIR", "artifacts/fast_path"))
    store.snapshot = SearchSnapshot(bundle, data_dir)
    classifier = HybridTriage(gemini, bundle)
    config = replace(DEFAULT_CONFIG, retrieval_backend="dense")
    detect_language_langdetect(MaskedTicket("", "Warm up the language profiles at startup."))

    def retrieve(ticket, k):
        matches = store.retrieve(ticket, k)
        for match in matches:
            row = gemini.fit.loc[match.ticket_id]
            match.body = mask_pii(row["body"]).text
            match.type = row["type"]
            match.queue = row["queue"]
            match.priority = row["priority"]
        return matches

    def decide(ticket, language, triage, precedents):
        result = gate(ticket, language, triage, precedents, config=config)
        return Decision.model_validate(asdict(result))

    def generate(ticket, language, precedents):
        result = draft_template(ticket, language, precedents, config=config)
        return Draft.model_validate(asdict(result))

    def verify(ticket, draft, language, precedents):
        result = validate(draft, language, precedents, ticket=ticket, config=config)
        return Validation.model_validate(asdict(result))

    return Components(
        mask=mask_ticket,
        detect_language=detect_language_langdetect,
        triage=None,
        retrieve=retrieve,
        gate=decide,
        draft=generate,
        validate=None,
        release_id=os.environ.get("SENTINELDESK_RELEASE_ID", "managed-triage"),
        retrieval_mode="agent_retrieval",
        generator_mode="offline_templates",
        artifacts={"training_examples": True, "drafting": True},
        triage_with_precedents=classifier.predict,
        validate_with_ticket=verify,
    )
