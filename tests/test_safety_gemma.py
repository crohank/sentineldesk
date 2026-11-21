from safety_fixtures import precedent, ticket

from sentineldesk.safety import contracts as c
from sentineldesk.safety import gemma
from sentineldesk.safety.drafting import draft_with_mode

T = ticket("The export to PDF fails with an error on the reports page.")
SPAN = "Could you please provide the exact error message and the version of the software you are using?"


def fake_calls(monkeypatch, responses):
    calls = iter(responses)

    def _call(model, prompt, timeout, json_mode=False):
        r = next(calls)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(gemma, "call", _call)


def test_timeout_falls_back_to_templates(monkeypatch):
    fake_calls(monkeypatch, [gemma.GemmaError("TimeoutError")] * 2)
    d, mode = draft_with_mode(T, "en", [precedent()], generator=c.GEMMA)
    assert mode == c.OFFLINE_TEMPLATES and not d.abstain and d.cited_ticket_ids == ["TKT-01234"]


def test_malformed_json_falls_back_after_one_repair(monkeypatch):
    fake_calls(monkeypatch, ["not json", "still not json"])
    assert draft_with_mode(T, "en", [precedent()], generator=c.GEMMA)[1] == c.OFFLINE_TEMPLATES


def test_valid_gemma_draft_with_supported_claim_is_used(monkeypatch):
    body = ('{"sentences": [{"text": "Could you please provide the exact error message and the version of the '
            'software you are using?", "ticket_id": "TKT-01234", "evidence": "' + SPAN + '"}], "abstain": false}')
    fake_calls(monkeypatch, [body, '{"grades": [{"i": 0, "grade": "supported"}]}'])
    d, mode = draft_with_mode(T, "en", [precedent()], generator=c.GEMMA)
    assert mode == c.GEMMA and d.claims[0].evidence_span == SPAN


def test_unsupported_number_rejects_the_gemma_draft(monkeypatch):
    body = ('{"sentences": [{"text": "We will fix the export within 24 hours.", "ticket_id": "TKT-01234", '
            '"evidence": "' + SPAN + '"}], "abstain": false}')
    fake_calls(monkeypatch, [body, body])
    assert draft_with_mode(T, "en", [precedent()], generator=c.GEMMA)[1] == c.OFFLINE_TEMPLATES


def test_parse_json_extracts_the_object():
    assert gemma.parse_json('{"ok": true}') == {"ok": True}
