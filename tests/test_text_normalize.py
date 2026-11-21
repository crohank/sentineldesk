from pathlib import Path

import pytest

from scripts.text_normalize import find_placeholders, normalize_text, ticket_text

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data/processed/kb_train.parquet"
TEST_INPUTS = ROOT / "data/processed/test_inputs.parquet"

EN_BODY = (
    "Dear Customer Support Team,\n\nI hope this message finds you well. I am writing to report that "
    "the billing dashboard shows duplicate charges for March.\n\nThank you for your assistance.\n\n"
    "Best regards,\n[Your Name]"
)
DE_BODY = (
    "Sehr geehrtes Support-Team,\n\nich hoffe, diese Nachricht erreicht Sie wohlbehalten. Seit dem Update "
    "startet die App nicht mehr.\n\nVielen Dank für Ihre Unterstützung.\n\nMit freundlichen Grüßen\n[Ihr Name]"
)


def test_none_and_empty():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""
    assert normalize_text("   \n\n ") == ""
    assert normalize_text(None, strip_greeting=True, placeholders="remove") == ""


def test_html_tags_and_entities():
    assert normalize_text("Hello<br>world<br/>again<br />end") == "Hello\nworld\nagain\nend"
    assert normalize_text("<p>One</p><p>Two</p>") == "One\n\nTwo"
    assert normalize_text("a&nbsp;b &amp; c") == "a b & c"
    assert normalize_text("<b>bold</b> text") == "bold text"


def test_html_step_keeps_placeholders():
    assert normalize_text("Call <tel_num> or ask <name>.<br>Thanks") == "Call <tel_num> or ask <name>.\nThanks"


def test_nfc():
    decomposed = "Grüße"
    assert normalize_text(decomposed) == "Grüße"


def test_whitespace_collapse():
    assert normalize_text("a  \t b") == "a b"
    assert normalize_text("a\n\n\n\nb") == "a\n\nb"
    assert normalize_text("  a \n b  ") == "a\nb"
    assert normalize_text("a\r\nb") == "a\nb"


def test_english_greeting_and_signoff_stripped():
    out = normalize_text(EN_BODY, strip_greeting=True)
    assert out == "I am writing to report that the billing dashboard shows duplicate charges for March."


def test_german_greeting_and_signoff_stripped():
    out = normalize_text(DE_BODY, strip_greeting=True)
    assert out == "Seit dem Update startet die App nicht mehr."


def test_inline_salutation_and_pleasantry():
    text = "Dear Customer Support, I hope this message reaches you in good health. The VPN drops every hour."
    assert normalize_text(text, strip_greeting=True) == "The VPN drops every hour."


def test_bare_addressee_salutation():
    assert normalize_text("Customer Support,<br>Login fails on Android.", strip_greeting=True) == "Login fails on Android."


@pytest.mark.parametrize("closing", [
    "Thank you.",
    "Thank you for your time and assistance.",
    "I look forward to hearing back soon.",
    "Thank you for your assistance, and I look forward to hearing from you soon.",
    "Vielen Dank im Voraus.",
    "Ich freue mich auf Ihre baldige Rückmeldung.",
    "Best regards, John Smith",
    "Mit freundlichen Grüßen,\nMax Mustermann",
])
def test_signoff_variants(closing):
    text = f"The export to CSV fails with error 500. {closing}"
    assert normalize_text(text, strip_greeting=True) == "The export to CSV fails with error 500."


def test_content_sentences_not_stripped():
    text = (
        "The invoice export crashes when the date range spans two years. "
        "Best practices for archiving would also help. "
        "Thank you, the workaround from last week no longer works"
    )
    assert normalize_text(text, strip_greeting=True) == text


def test_message_starting_with_hi_and_content_kept():
    text = "Hi I cannot log in since the update\nPlease reset my account"
    assert normalize_text(text, strip_greeting=True) == text


def test_customer_support_as_subject_not_salutation():
    text = "Customer Support is reaching out to inquire about data retention."
    assert normalize_text(text, strip_greeting=True) == text


def test_never_empty_when_all_boilerplate():
    for text in ["Dear Customer Support,\n\nThank you.", "Thank you!", "Sehr geehrte Damen und Herren,", "Hello"]:
        out = normalize_text(text, strip_greeting=True)
        assert out, text
        assert out == normalize_text(text)


def test_placeholder_modes():
    text = "Call [Tel Nummer] or <Tel-Nummer>, ask [Your Name] or <Name>, account [Konto-Nr.], mail <email>, <website_url>, <amount>."
    assert normalize_text(text) == text
    assert normalize_text(text, placeholders="canonical") == (
        "Call <tel_num> or <tel_num>, ask <name> or <name>, account <acc_num>, mail <email>, <url>, <amount>."
    )
    assert normalize_text(text, placeholders="remove") == "Call or, ask or, account, mail,,."


def test_unknown_brackets_untouched():
    text = "See [Company Name] and <company_name> on [date]."
    assert normalize_text(text, placeholders="canonical") == text
    assert normalize_text(text, placeholders="remove") == text


def test_bad_placeholder_mode():
    with pytest.raises(ValueError):
        normalize_text("x", placeholders="drop")


def test_find_placeholders():
    assert find_placeholders("Reach me at [Telefonnummer], <name>, [acc_num] or [Phone Number].") == [
        "acc_num", "name", "tel_num",
    ]
    assert find_placeholders("no tokens [Company] here") == []
    assert find_placeholders(None) == []


@pytest.mark.parametrize("kw", [
    {},
    {"strip_greeting": True},
    {"placeholders": "canonical"},
    {"strip_greeting": True, "placeholders": "remove"},
])
@pytest.mark.parametrize("text", [
    EN_BODY, DE_BODY, "a&amp;nbsp;b", "x <br><br><br> y", "[Account-Nummer: <acc_num>] ok", "Thank you.",
    "Customer Support, Hello, I hope this message finds you well. Printer jams.",
])
def test_idempotent(text, kw):
    once = normalize_text(text, **kw)
    assert normalize_text(once, **kw) == once


def test_ticket_text():
    assert ticket_text("VPN down", "Dear Support Team,\nIt drops hourly.", strip_greeting=True) == "VPN down\n\nIt drops hourly."
    assert ticket_text("", "Body only") == "Body only"
    assert ticket_text(None, None) == ""
    assert ticket_text("Thank you", "Body", strip_greeting=True) == "Thank you\n\nBody"
    assert ticket_text("Call <Tel-Nummer>", "x", placeholders="canonical") == "Call <tel_num>\n\nx"


@pytest.mark.skipif(not (KB.exists() and TEST_INPUTS.exists()), reason="processed data absent")
def test_strip_greeting_never_empties_real_bodies():
    pd = pytest.importorskip("pandas")
    bodies = pd.concat([
        pd.read_parquet(KB, columns=["body"])["body"],
        pd.read_parquet(TEST_INPUTS, columns=["body"])["body"],
    ])
    empty = [b for b in bodies if b.strip() and not normalize_text(b, strip_greeting=True)]
    assert empty == []
