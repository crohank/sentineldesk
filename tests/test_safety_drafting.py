from safety_fixtures import ANSWER_EN, precedent, ticket

from sentineldesk.safety import contracts as c
from sentineldesk.safety.drafting import draft, draft_with_mode
from sentineldesk.safety.validation import validate

ROUTINE = ticket("The export to PDF fails with an error on the reports page.")


def test_routine_english_draft_cites_its_evidence():
    p = precedent()
    d = draft(ROUTINE, "en", [p])
    assert not d.abstain and d.cited_ticket_ids == ["TKT-01234"]
    assert "[TKT-01234]" in d.reply_text and d.reply_text.startswith("Hello,")
    assert d.claims and all(cl.evidence_span in p.answer_snippet for cl in d.claims)
    v = validate(d, "en", [p])
    assert v.valid, v.errors


def test_german_reply_from_english_evidence():
    p = precedent()
    d = draft(ticket("Der PDF-Export auf der Berichtsseite schlägt mit einem Fehler fehl."), "de", [p])
    assert d.reply_text.startswith("Guten Tag,") and d.cited_ticket_ids == ["TKT-01234"]
    assert all(cl.evidence_span in p.answer_snippet for cl in d.claims)
    assert validate(d, "de", [p]).valid


def test_german_evidence_is_matched():
    p = precedent(answer="Bitte senden Sie uns die genaue Fehlermeldung. Starten Sie das Gerät neu.", lang="de")
    d = draft(ticket("Fehler beim Export."), "de", [p])
    assert [cl.text for cl in d.claims] == [
        "Könnten Sie uns die genaue Fehlermeldung schicken?"
    ]


def test_troubleshooting_steps_are_never_suggested():
    p = precedent(answer="Please restart the application and send us the exact error message.")
    d = draft(ticket("The app keeps showing an error."), "en", [p])
    assert not any("restart" in cl.text.lower() for cl in d.claims)
    assert any("error message" in cl.text for cl in d.claims)


def test_no_supported_action_abstains():
    d = draft(ROUTINE, "en", [precedent(answer="We have forwarded your request to the billing team.")])
    assert d.abstain and not d.claims
    assert validate(d, "en", []).errors == ["draft abstained"]


def test_low_score_precedents_are_not_used():
    assert draft(ROUTINE, "en", [precedent(score=0.05)]).abstain


def test_placeholders_never_reach_the_reply():
    p = precedent(answer="Please share your account number <acc_num> and call us at <tel_num> at a convenient time.")
    d = draft(ROUTINE, "en", [p])
    assert not d.abstain and "<" not in d.reply_text
    assert validate(d, "en", [p]).valid


def test_at_most_three_actions():
    p = precedent(answer=(
        "Could you provide the error message? Please send more details. Please confirm the browser version. "
        "Please share your account number. Please tell us a convenient time for a call."
    ))
    d = draft(ROUTINE, "en", [p])
    assert len(d.claims) == 3 and len(d.cited_ticket_ids) == 1


def test_unsupported_language_abstains_and_mode_is_reported():
    d, mode = draft_with_mode(ROUTINE, "fr", [precedent()])
    assert d.abstain and mode == c.OFFLINE_TEMPLATES
    assert ANSWER_EN  # fixture text is shared with the gate tests
