"""Generation checks. Run: python -m pytest tests/test_generation.py (no model needed, except the last test)."""
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.generation import (  # noqa: E402
    ABSTAIN, Claim, Draft, Generator, assemble, check_draft, extractive_draft, fill_placeholders,
)


@dataclass
class P:  # same attribute names as D3's and D5's Precedent
    ticket_id: str
    score: float
    subject: str
    answer_snippet: str
    answer_language: str


EN = [
    P("TKT-00100", 0.52, "Printer prints blank pages",
      "Thank you for contacting us, <name>. Please check whether the printer cartridge is installed correctly "
      "and run the printhead cleaning from the printer menu. If the issue persists, send us your account number <acc_num>.", "en"),
    P("TKT-00200", 0.48, "Printer offline",
      "We received your report about the office printer. Could you please tell us the printer model and any error "
      "message shown on its display?", "en"),
]
DE = [
    P("TKT-00300", 0.50, "Drucker druckt leere Seiten",
      "Vielen Dank für Ihre Nachricht. Bitte prüfen Sie, ob die Druckerpatrone richtig eingesetzt ist, und starten "
      "Sie die Druckkopfreinigung im Druckermenü.", "de"),
]
TICKET = {"subject": None, "body": "Our office printer only prints blank pages since yesterday."}


def good_draft() -> Draft:
    span = "Please check whether the printer cartridge is installed correctly"
    return Draft(
        reply_text=f"Thank you for reaching out. Please check that the printer cartridge is installed correctly. [TKT-00100]",
        cited_ticket_ids=["TKT-00100"],
        claims=[Claim("Please check that the printer cartridge is installed correctly.", "TKT-00100", span)],
        abstain=False)


def test_good_draft_passes():
    assert check_draft(good_draft(), "en", EN, TICKET["body"]) == []


def test_abstain_passes():
    assert check_draft(ABSTAIN, "en", EN) == []


def test_fabricated_id_fails():
    d = good_draft()
    d.reply_text += " Also restart it. [TKT-99999]"
    assert any("TKT-99999" in e for e in check_draft(d, "en", EN))


def test_non_verbatim_span_fails():
    d = good_draft()
    d.claims[0].evidence_span = "Please verify the cartridge is seated properly in the printer"
    assert any("not an exact quote" in e for e in check_draft(d, "en", EN))


def test_invented_number_fails():
    d = good_draft()
    d.reply_text = d.reply_text.replace("correctly.", "correctly within 48 hours.")
    assert any("number 48" in e for e in check_draft(d, "en", EN))


def test_new_commitment_fails():
    d = good_draft()
    d.reply_text = d.reply_text.replace("Thank you for reaching out.", "We guarantee a full refund.")
    errors = check_draft(d, "en", EN)
    assert any("guarantee" in e for e in errors) and any("refund" in e for e in errors)


def test_placeholder_fails():
    d = good_draft()
    d.reply_text = d.reply_text.replace("Thank you", "Thank you <name>")
    assert any("placeholder" in e for e in check_draft(d, "en", EN))


def test_wrong_language_fails():
    d = good_draft()
    assert any("expected 'de'" in e for e in check_draft(d, "de", EN))


def test_assemble_markers_placeholders_greeting_and_reassign():
    raw = {"abstain": False, "sentences": [
        {"text": "Thank you for your message.", "source": "TKT-00200",
         "evidence_span": "We received your report about the office printer."},
        # quote really comes from TKT-00100 although the model named TKT-00200
        {"text": "Please send us your account number <acc_num>.", "source": "TKT-00200",
         "evidence_span": "run the printhead cleaning from the printer menu"},
    ]}
    d = assemble(raw, EN, "en")
    assert d.reply_text.startswith("Thank you for your message. Please send us your account number. [TKT-00100]")
    assert "[TKT-00100]" in d.reply_text and "[TKT-00200]" not in d.reply_text  # greeting not cited, source fixed
    assert d.claims[0].ticket_id == "TKT-00100" and d.cited_ticket_ids == ["TKT-00100"]


def test_fill_placeholders_both_languages():
    assert fill_placeholders("Please allow us to call <tel_num>, <name>.", "en") == "Please allow us to call <tel_num>."  # ambiguous: left for repair
    assert fill_placeholders("Send your account number <acc_num> and contact us at <tel_num>.", "en") == \
        "Send your account number and contact us by phone."
    assert fill_placeholders("Kontaktieren Sie uns unter <tel_num>.", "de") == "Kontaktieren Sie uns telefonisch."
    assert fill_placeholders("Senden Sie [acc_num].", "de") == "Senden Sie Ihre Kontonummer."


def test_extractive_fallback_passes_or_abstains():
    for lang, precs in [("en", EN), ("de", DE)]:
        ticket = TICKET if lang == "en" else {"subject": None, "body": "Unser Drucker druckt nur leere Seiten, die Druckerpatrone ist neu."}
        d = extractive_draft(ticket, lang, precs)
        assert d.abstain or check_draft(d, lang, precs, ticket["body"]) == []
    off_topic = extractive_draft({"subject": None, "body": "What is the weather in Paris tomorrow?"}, "en", EN)
    assert off_topic.abstain


def test_relevance_guard_and_unsupported_language():
    gen = Generator([])
    assert gen.draft_with_mode(TICKET, "en", EN, min_top_score=0.9).draft.abstain
    assert gen.draft_with_mode(TICKET, "fr", EN).mode == "none"
    res = gen.draft_with_mode(TICKET, "en", EN)
    assert res.mode in ("offline_templates", "none")


def test_gemma_end_to_end():
    from sentineldesk.dense import ollama_available
    from sentineldesk.generation import GemmaOllama
    if not ollama_available():
        pytest.skip("Ollama not running")
    res = Generator([GemmaOllama()], budget_s=15).draft_with_mode(TICKET, "en", EN)
    assert res.draft.abstain or check_draft(res.draft, "en", EN, TICKET["body"]) == []


def test_junk_sentence_fails():
    d = good_draft()
    d.claims[0].text = "source: TKT-00100"
    assert any("not a reply sentence" in e for e in check_draft(d, "en", EN))


def test_ambiguous_phone_placeholder_is_not_guessed():
    raw = {"abstain": False, "sentences": [
        {"text": "Please allow us to call <tel_num> to discuss further.", "source": "TKT-00200",
         "evidence_span": "Could you please tell us the printer model"}]}
    d = assemble(raw, EN, "en")
    assert "<tel_num>" in d.reply_text and any("placeholder" in e for e in check_draft(d, "en", EN))
