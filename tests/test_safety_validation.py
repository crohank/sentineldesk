from safety_fixtures import precedent, ticket

from sentineldesk.safety.config import SafetyConfig
from sentineldesk.safety.contracts import Claim, Draft
from sentineldesk.safety.drafting import draft
from sentineldesk.safety.validation import validate

ROUTINE = ticket("The export to PDF fails with an error on the reports page.")
SLA_ANSWER = "We usually reply within 48 hours. We reply in up to 48 hours during holidays."


def template_draft():
    p = precedent()
    return draft(ROUTINE, "en", [p]), p


def model_draft(sentence: str, span: str) -> Draft:
    return Draft(
        reply_text=f"Hello,\n{sentence} [TKT-01234]\nBest regards,\nSupport Team",
        cited_ticket_ids=["TKT-01234"],
        claims=[Claim(text=sentence, ticket_id="TKT-01234", evidence_span=span)],
        abstain=False,
    )


def errors(d, language="en", precs=None, **kw):
    return validate(d, language, [precedent(answer=SLA_ANSWER)] if precs is None else precs, **kw).errors


def test_template_draft_passes():
    d, p = template_draft()
    assert validate(d, "en", [p]).valid


def test_fabricated_citation_is_caught():
    d, p = template_draft()
    d.cited_ticket_ids.append("TKT-99999")
    assert "cited TKT-99999 was not retrieved" in validate(d, "en", [p]).errors


def test_span_not_in_answer_is_caught():
    d, p = template_draft()
    d.claims[0].evidence_span = "We guarantee a fix within 48 hours."
    assert any("evidence span not found" in e for e in validate(d, "en", [p]).errors)


def test_changed_number_is_caught():
    d = model_draft("We will reply within 24 hours.", "We usually reply within 48 hours.")
    assert "number not in evidence: 24" in errors(d)


def test_missing_qualifier_is_caught():
    d = model_draft("We guarantee a reply within 48 hours.", "We reply in up to 48 hours during holidays.")
    e = errors(d)
    assert "qualifier not in evidence: guarantee" in e and "qualifier not in evidence: within" in e
    assert any(x.startswith("commitment not in evidence") for x in e)


def test_supported_model_sentence_passes_these_checks():
    d = model_draft("We usually reply within 48 hours.", "We usually reply within 48 hours.")
    assert errors(d) == []


def test_unsupported_sentence_is_caught():
    d = model_draft("We usually reply within 48 hours.", "We usually reply within 48 hours.")
    d.reply_text = d.reply_text.replace("Best regards", "Your refund is on its way.\nBest regards")
    assert any(e.startswith("unsupported sentence: Your refund") for e in errors(d))


def test_wrong_language_is_caught():
    d, p = template_draft()
    assert "template text is not the reviewed de version" in validate(d, "de", [p]).errors
    m = model_draft("We usually reply within 48 hours.", "We usually reply within 48 hours.")
    assert any("does not match de" in e for e in errors(m, "de"))


def test_placeholder_in_reply_is_caught():
    d = model_draft("We usually reply within 48 hours.", "We usually reply within 48 hours.")
    d.reply_text = d.reply_text.replace("Hello,", "Hello <name>,")
    assert any(e.startswith("placeholder in reply") for e in errors(d))


def test_unknown_product_name_in_model_draft_is_caught():
    d = model_draft("We usually reply within 48 hours via SAPConnect.", "We usually reply within 48 hours.")
    assert any(e.startswith("names not in ticket or evidence: SAPConnect") for e in errors(d, ticket=ROUTINE))


def test_too_long_is_caught():
    d, p = template_draft()
    assert any(e.startswith("reply too long") for e in validate(d, "en", [p], config=SafetyConfig(max_reply_words=5)).errors)
