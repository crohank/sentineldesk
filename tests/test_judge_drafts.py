import json

import pandas as pd
import pytest

from scripts import judge_drafts as jd

PROC = jd.PROC
needs_data = pytest.mark.skipif(
    not (PROC / "kb_train.parquet").exists() or not jd.SAMPLE.exists(), reason="split or sample missing")


@pytest.fixture(scope="module")
def kb():
    return pd.read_parquet(PROC / "kb_train.parquet", columns=["ticket_id", "answer"])


def fake_predictions(kb, sample, tmp_path):
    clean = kb[(kb.answer.str.len() > 80) & ~kb.answer.str.contains(jd.PLACEHOLDER)
               & ~kb.answer.str[:60].str.contains(r"\d")]
    good, other = clean.iloc[0], clean.iloc[1]
    span = good.answer[:60]
    rows = [
        {"ticket_id": sample[0], "draft_text": f"{span} See {good.ticket_id}.", "cited_ids": [good.ticket_id],
         "retrieved_ids": [good.ticket_id, other.ticket_id],
         "draft_claims": [{"text": span, "ticket_id": good.ticket_id, "evidence_span": span}]},
        {"ticket_id": sample[1], "draft_text": "We guarantee a fix within 48 hours.", "cited_ids": [good.ticket_id],
         "draft_claims": [{"text": "We guarantee a fix within 48 hours.", "ticket_id": good.ticket_id,
                           "evidence_span": "guaranteed within 48 hours"}]},
        {"ticket_id": sample[2], "draft_text": "", "draft_abstain": True},
        {"ticket_id": sample[3], "draft_text": "Please call <tel_num>.", "cited_ids": [other.ticket_id]},
        {"ticket_id": sample[4], "draft_text": "Restart the router.", "cited_ids": [other.ticket_id],
         "retrieved_ids": [good.ticket_id]},
    ]
    path = tmp_path / "preds.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


@needs_data
def test_mock_run_reports_checks(kb, tmp_path, monkeypatch):
    monkeypatch.setattr(jd, "REPORTS", tmp_path / "reports")
    (tmp_path / "reports").mkdir()
    sample = [l for l in jd.SAMPLE.read_text().split() if l]
    r = jd.judge_run(fake_predictions(kb, sample, tmp_path), "t", "mock", None)
    s = r["summary"]
    assert s["drafts_emitted"] == 4 and s["abstentions"] == 1
    assert s["sample_ids_in_file"] == 5
    assert s["citation_valid"] == "0.750 (3/4)"  # last draft cites an ID it never retrieved
    assert s["drafts_with_unsupported_numbers"].endswith("(1/4)")
    assert s["drafts_with_placeholders"].endswith("(1/4)")
    assert s["evidence_spans_found"] == "0.500 (1/2)"
    md = (tmp_path / "reports" / "t" / "generation_judge.md").read_text(encoding="utf-8")
    assert "mock judge is rule based" in md


@needs_data
def test_holdout_citation_is_rejected(kb, tmp_path, monkeypatch):
    monkeypatch.setattr(jd, "REPORTS", tmp_path / "reports")
    sample = [l for l in jd.SAMPLE.read_text().split() if l]
    path = tmp_path / "leak.jsonl"
    path.write_text(json.dumps({"ticket_id": sample[0], "draft_text": "x.", "cited_ids": [sample[1]]}), encoding="utf-8")
    with pytest.raises(jd.JudgeError, match="holdout"):
        jd.judge_run(path, "t", "mock", None)


@needs_data
def test_manual_sheet_round_trip(kb, tmp_path, monkeypatch):
    monkeypatch.setattr(jd, "REPORTS", tmp_path / "reports")
    sample = [l for l in jd.SAMPLE.read_text().split() if l]
    preds = fake_predictions(kb, sample, tmp_path)
    jd.judge_run(preds, "m", "manual", None)
    sheet_path = tmp_path / "reports" / "m" / "judge_sheet.csv"
    sheet = pd.read_csv(sheet_path)
    sheet["verdict"] = "supported"
    sheet.loc[0, "verdict"] = "unsupported"
    sheet.to_csv(sheet_path, index=False)
    r = jd.judge_run(preds, "m", "manual", None, from_sheet=sheet_path)
    assert r["summary"]["claims_graded"] == len(sheet)
    assert r["meta"]["judge"].startswith("manual")


def test_remote_backend_needs_model():
    with pytest.raises(jd.JudgeError, match="--model"):
        jd.judge_run(jd.SAMPLE, "x", "gemini", None)


def test_number_check_ignores_ticket_ids():
    assert jd.NUMBER.findall(jd.TICKET_ID.sub(" ", "See TKT-01234 for 48 hours")) == ["48"]


def test_placeholder_patterns():
    found = jd.PLACEHOLDER.findall("Call <tel_num> or [Ihr Name], account [acc_num]. Not [Step 1].")
    assert found == ["<tel_num>", "[Ihr Name]", "[acc_num]"]
