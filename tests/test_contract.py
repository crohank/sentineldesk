"""Contract tests: fixtures, schema snapshot, API behaviour, citation integrity."""
import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from sentineldesk.api import create_app
from sentineldesk.components import FIXTURE_NAMES, fixture_components, load_fixture
from sentineldesk.pipeline import analyze, citation_errors
from sentineldesk.schemas import SCHEMA_PATH, Claim, TicketInput, contract_json_schema

KB_LAST_ID = 22856  # TKT-22856 is the last KB row (D1-04); anything above is holdout.


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(fixture_components()))


# ------------------------------------------------------------------ fixtures


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_validates(name):
    load_fixture(name)  # raises on schema mismatch


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_ids_are_kb_range(name):
    r = load_fixture(name)
    ids = [p.ticket_id for p in r.precedents]
    if r.draft:
        ids += r.draft.cited_ticket_ids + [c.ticket_id for c in r.draft.claims]
    assert all(int(i.split("-")[1]) <= KB_LAST_ID for i in ids), ids


def test_routine_fixture_citations_are_consistent():
    r = load_fixture("routine")
    assert r.decision.status == "draft_ready" and r.draft is not None
    assert citation_errors(r.draft, r.precedents) == []
    assert len(r.precedents) == 5


def test_on_call_fixture_shape():
    r = load_fixture("on_call")
    assert r.decision.status == "escalate" and r.decision.destination == "on_call"
    assert r.draft is None


def test_insufficient_fixture_shape():
    r = load_fixture("insufficient")
    assert r.triage is None and r.precedents == [] and r.draft is None
    assert [x.code for x in r.decision.reasons] == ["INSUFFICIENT_INPUT"]


@pytest.mark.parametrize("name", ["routine", "on_call"])
def test_fixture_probabilities_sum_to_one(name):
    t = load_fixture(name).triage
    for head in (t.type, t.queue, t.priority):
        assert sum(head.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
        assert head.probabilities[head.label] == pytest.approx(head.confidence)


def test_schema_snapshot_is_current():
    committed = json.loads(SCHEMA_PATH.read_text())
    assert committed == contract_json_schema(), "schemas changed: run `uv run export_schemas` and commit"


# ------------------------------------------------------------------ citation check


def test_citation_check_catches_fabricated_and_unquoted():
    r = load_fixture("routine")
    bad = r.draft.model_copy(
        update={
            "cited_ticket_ids": ["TKT-00001", "TKT-99999"],
            "claims": [Claim(text="x", ticket_id="TKT-00001", evidence_span="guaranteed within 48 hours")],
        }
    )
    errors = citation_errors(bad, r.precedents)
    assert any("TKT-99999" in e for e in errors)
    assert any("evidence span not found" in e for e in errors)


# ------------------------------------------------------------------ pipeline


def _boom(*args):
    raise TimeoutError("provider timed out on: secret ticket text")


@pytest.mark.parametrize("stage", ["draft", "validate"])
def test_draft_stage_exception_becomes_review(stage, caplog):
    comps = dataclasses.replace(fixture_components(), **{stage: _boom})
    r = analyze(TicketInput(body="How do I reset my password?"), comps, mode="batch")
    assert r.decision.status == "review"
    assert [x.code for x in r.decision.reasons] == ["DRAFT_FAILED"]
    assert r.draft is None
    assert r.triage is not None and len(r.precedents) == 5  # evidence kept for the operator
    assert "secret ticket text" not in caplog.text


def test_fabricated_citation_draft_is_withheld():
    good = load_fixture("routine").draft
    bad = good.model_copy(update={"cited_ticket_ids": good.cited_ticket_ids + ["TKT-09999"]})
    comps = dataclasses.replace(fixture_components(), draft=lambda *a: bad)
    r = analyze(TicketInput(body="How do I reset my password?"), comps)
    assert r.draft is None and r.decision.reasons[0].code == "DRAFT_FAILED"


def test_analyze_default_components():
    r = analyze(TicketInput(body="How do I reset my password?"), mode="batch")
    assert r.retrieval_mode == "fixture"


# ------------------------------------------------------------------ API


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["retrieval_mode"] == "fixture" and body["generator_mode"] == "fixture"
    assert body["status"] == "degraded"  # no real artifacts yet


def test_triage_routine(client):
    r = client.post("/api/triage", json={"subject": "Password", "body": "I forgot my password. How do I reset it?"})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"]["status"] == "draft_ready"
    assert body["draft"]["cited_ticket_ids"]
    assert body["stage_ms"]["triage"] is not None


def test_null_subject_is_valid(client):
    r = client.post("/api/triage", json={"subject": None, "body": "My invoice is missing."})
    assert r.status_code == 200


def test_missing_subject_key_is_valid(client):
    r = client.post("/api/triage", json={"body": "My invoice is missing."})
    assert r.status_code == 200


@pytest.mark.parametrize("body", ["", "   ", "?!...", "\n\t"])
def test_empty_body_needs_clarification(client, body):
    r = client.post("/api/triage", json={"subject": "Help", "body": body})
    assert r.status_code == 200
    out = r.json()
    assert out["decision"]["status"] == "needs_clarification"
    assert out["triage"] is None and out["precedents"] == [] and out["draft"] is None
    assert out["stage_ms"]["triage"] is None


def test_extra_field_rejected(client):
    r = client.post("/api/triage", json={"body": "hi", "queue": "Billing and Payments"})
    assert r.status_code == 422


def test_missing_body_rejected(client):
    assert client.post("/api/triage", json={"subject": "x"}).status_code == 422


def test_german_detected(client):
    r = client.post(
        "/api/triage",
        json={"body": "Ich habe mein Passwort vergessen und habe noch Zugriff auf meine registrierte E-Mail-Adresse."},
    )
    assert r.json()["language"] == "de"


# ------------------------------------------------------------------ console


def test_console_get(client):
    r = client.get("/")
    assert r.status_code == 200 and "<form" in r.text


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_console_renders_fixture(client, name):
    r = client.get(f"/?fixture={name}")
    assert r.status_code == 200
    assert "Showing stored fixture" in r.text


def test_console_links_citations_to_evidence(client):
    html = client.get("/?fixture=routine").text
    assert 'href="#ev-TKT-00001"' in html and 'id="ev-TKT-00001"' in html
    assert "<mark>" in html


def test_console_post_escapes_input(client):
    r = client.post("/", data={"subject": "", "body": "<script>alert(1)</script> password reset"})
    assert r.status_code == 200
    assert "<script>alert(1)</script>" not in r.text


def test_console_has_no_external_assets(client):
    """Offline demo: every stylesheet, script and image must be local and relative."""
    import re

    html = client.get("/?fixture=routine").text
    asset_urls = re.findall(r'<(?:link|script|img)[^>]*(?:href|src)="([^"]+)"', html)
    assert asset_urls, "expected the local stylesheet and script"
    assert all(u.startswith("/static/") for u in asset_urls), asset_urls


@pytest.mark.parametrize("key", ["outage-de", "password-en", "refund-en", "override-en", "password-de"])
def test_sample_prefills_form_without_result(client, key):
    from sentineldesk.api import SAMPLES_BY_KEY

    html = client.get(f"/?sample={key}").text
    assert SAMPLES_BY_KEY[key].body.split()[0] in html
    assert "has-result" not in html  # a sample never shows a result by itself


def test_unknown_sample_404(client):
    assert client.get("/?sample=nope").status_code == 404


def test_result_shows_pipeline_stepper(client):
    html = client.get("/?fixture=insufficient").text
    assert 'class="stepper"' in html and "skipped" in html
