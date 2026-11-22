"""Shared validation helpers for comparing triage variants on identical partitions.

Mirrors the procedure in scripts/train_triage.py (left untouched): same val halves,
same 10-bin ECE, same rule-6 grid and selection.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
OUTAGE_GRID = np.round(np.arange(0.30, 0.701, 0.05), 2)
OUTAGE_RECALL_TARGET = 0.90


def val_halves(val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(cal, thr) exactly as in train_triage.py."""
    strat = val["type"] + "|" + val["priority"]
    cal, thr = train_test_split(val, test_size=0.5, random_state=SEED, stratify=strat)
    return cal, thr


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    rows, total = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            total += m.mean() * gap
            rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                         "conf": float(conf[m].mean()), "acc": float(correct[m].mean())})
    return float(total), rows


def outage_search(p_type: pd.DataFrame, p_priority: pd.DataFrame, gold: np.ndarray) -> dict:
    """Classifier joint prediction (gate rule 2) OR probability rule (gate rule 6), grid-searched."""
    cls_hit = ((p_type.idxmax(axis=1) == "Incident") & (p_priority.idxmax(axis=1) == "high")).to_numpy()
    p_inc = p_type["Incident"].to_numpy()
    p_high = p_priority["high"].to_numpy()
    grid = []
    for a in OUTAGE_GRID:
        for b in OUTAGE_GRID:
            fire = cls_hit | ((p_inc >= a) & (p_high >= b))
            grid.append({"p_incident": float(a), "p_high": float(b), "recall": float(fire[gold].mean()),
                         "precision": float(gold[fire].mean()), "escalation_rate": float(fire.mean())})
    ok = [g for g in grid if g["recall"] >= OUTAGE_RECALL_TARGET]
    if ok:
        chosen, status = max(ok, key=lambda g: (g["precision"], g["recall"])), "met"
    else:
        chosen, status = max(grid, key=lambda g: (g["recall"], g["precision"])), "not met (best attainable recall)"
    return {
        "gold_n": int(gold.sum()),
        "classifier_recall": float(cls_hit[gold].mean()),
        "classifier_precision": float(gold[cls_hit].mean()) if cls_hit.any() else 0.0,
        "target_recall": OUTAGE_RECALL_TARGET, "target_status": status, "chosen": chosen, "grid": grid,
    }


def latency(predict_fn, rows: pd.DataFrame, n: int = 100, warmup: int = 5) -> dict:
    """Per-ticket wall time of predict_fn(subject, body) over n rows (after warmup)."""
    rows = rows.iloc[: n + warmup]
    times = []
    for i, (s, b) in enumerate(zip(rows["subject"], rows["body"])):
        t0 = time.perf_counter()
        predict_fn(s, b)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"n": len(t), "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean())}
