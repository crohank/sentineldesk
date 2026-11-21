"""D3 checks. Run: python -m pytest tests/test_retrieval.py (needs artifacts/retrieval/full)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.language import detect_language  # noqa: E402
from sentineldesk.retrieval import Retriever  # noqa: E402
from sentineldesk.textprep import clean_for_search  # noqa: E402

FULL = ROOT / "artifacts" / "retrieval" / "full"
FIT = ROOT / "artifacts" / "retrieval" / "fit"


@pytest.fixture(scope="module")
def full():
    if not FULL.exists():
        pytest.skip("run scripts/build_index.py --scope full first")
    return Retriever.load(FULL)


def test_index_has_only_kb_ids(full):
    test_ids = set(pd.read_parquet(ROOT / "data/processed/test_inputs.parquet", columns=["ticket_id"])["ticket_id"])
    assert not set(full.ids) & test_ids
    assert (full.ids < "TKT-22857").all()


def test_fit_index_excludes_validation():
    if not FIT.exists():
        pytest.skip("no fit index")
    parts = pd.read_csv(ROOT / "splits/kb_partitions_provisional.csv")
    held = set(parts.loc[parts["partition"] != "fit", "ticket_id"])
    assert not set(np.load(FIT / "ids.npy", allow_pickle=True)) & held


def test_null_subject_and_shape(full):
    res = full.retrieve({"subject": None, "body": "My VPN connection drops every few minutes since the update."}, k=5)
    assert len(res) == 5
    assert [p.score for p in res] == sorted((p.score for p in res), reverse=True)
    for p in res:
        assert p.ticket_id.startswith("TKT-") and p.answer_snippet and p.answer_language


def test_blank_body_returns_nothing(full):
    assert full.retrieve({"subject": None, "body": "   "}, k=5) == []


def test_object_input(full):
    class T:
        subject = "Rechnung"
        body = "Ich finde meine Rechnung vom letzten Monat nicht."
    assert len(full.retrieve(T(), k=3)) == 3


def test_boilerplate_removed():
    t = clean_for_search(None, "Dear Customer Support Team,\n\nMy printer prints blank pages.\n\nThank you.\n[Your Name]")
    assert t == "my printer prints blank pages."


@pytest.mark.parametrize("text,lang", [
    ("Ich habe mein Passwort vergessen", "de"),
    ("My VPN is down since this morning", "en"),
    ("Bonjour, je ne peux pas me connecter à mon compte", "fr"),
    ("Login broken", "und"),
    ("", "und"),
    (None, "und"),
    ("!!! ???", "und"),
])
def test_language(text, lang):
    assert detect_language(text)[0] == lang
