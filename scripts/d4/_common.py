"""Paths and loaders for D4's scripts. Reads only KB data and holdout inputs, never labels."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed"
DEV = DATA / "d4_dev"
REPORTS = ROOT / "reports" / "d4"

_DE = re.compile(r"\b(?:der|die|das|und|ist|nicht|sie|wir|bitte|mit|für|ich|ihre|uns|eine?n?|zu|auf|wird)\b", re.I)
_EN = re.compile(r"\b(?:the|and|is|not|you|we|please|with|for|i|your|us|an?|to|on|will)\b", re.I)


def guess_language(text: str) -> str:
    """Cheap en/de guess from function words, for counting answer languages."""
    de, en = len(_DE.findall(text or "")), len(_EN.findall(text or ""))
    return "de" if de > en else "en"


def load_kb():
    """KB rows with their D1 partition (fit / calib / val)."""
    import pandas as pd

    kb = pd.read_parquet(DATA / "kb_train.parquet")
    parts = pd.read_parquet(DATA / "kb_partitions.parquet")
    kb = kb.merge(parts, on="ticket_id", how="left")
    for col in ("subject", "body", "answer"):
        kb[col] = kb[col].fillna("")
    return kb
