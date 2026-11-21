"""Blend test: TF-IDF and Gemini probabilities averaged per head, on the 300 val tickets.

The pre-registered weight is w = 0.5 (Gemini share). Other weights are printed for
context only; choosing w from this table would tune on the evaluation tickets, so the
real weight is fit on `calib` later (docs/D1_decisions.md D1-47).

TF-IDF is fit on the `fit` partition, same recipe as baseline_tfidf.py. Gemini
probabilities come from data/predictions/val/<run>.parquet (llm_classify.py).
Writes reports/llm/blend_val300.md.

  python scripts/blend_val.py --runs val300_flash_few8 val300_lite_few8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.llm_classify import HEADS, PROC, PRED_DIR, REPORTS, load_labels, select_rows  # noqa: E402

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
PREREGISTERED = 0.5
ECE_BINS = 10


def tfidf_probs(pool: pd.DataFrame, queries: pd.DataFrame, labels: dict) -> dict[str, np.ndarray]:
    def text(df):
        return df["subject"] + " \n " + df["body"]
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([wv.fit_transform(text(pool)), cv.fit_transform(text(pool))]).tocsr()
    Xq = hstack([wv.transform(text(queries)), cv.transform(text(queries))]).tocsr()
    out = {}
    for h in HEADS:
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(Xtr, pool[h])
        p = pd.DataFrame(clf.predict_proba(Xq), columns=clf.classes_)
        out[h] = p.reindex(columns=labels[h], fill_value=0.0).to_numpy()
    return out


def llm_probs(run: str, ids: pd.Series, labels: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    df = pd.read_parquet(PRED_DIR / "val" / f"{run}.parquet").set_index("ticket_id").loc[ids]
    probs, stated = {}, {}
    for h in HEADS:
        m = np.array([[json.loads(s).get(l, 0.0) for l in labels[h]] for s in df[f"probs_{h}"]], dtype=float)
        sums = m.sum(axis=1, keepdims=True)
        # Rows where the model gave all zeros fall back to a one-hot on its stated label.
        onehot = (np.array(labels[h])[None, :] == df[f"pred_{h}"].to_numpy()[:, None]).astype(float)
        probs[h] = np.where(sums > 0, m / np.where(sums > 0, sums, 1), onehot)
        stated[h] = df[f"pred_{h}"].to_numpy()
    return probs, stated


def ece(conf: np.ndarray, correct: np.ndarray) -> float:
    bins = np.minimum((conf * ECE_BINS).astype(int), ECE_BINS - 1)
    return float(sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(ECE_BINS) if (bins == b).any()))


def evaluate(gold: pd.DataFrame, p: dict[str, np.ndarray], labels: dict) -> dict:
    out = {}
    for h in HEADS:
        pred = np.array(labels[h])[p[h].argmax(axis=1)]
        ok = pred == gold[h].to_numpy()
        out[h] = {"macro_f1": float(f1_score(gold[h], pred, average="macro")),
                  "acc": float(ok.mean()), "ece": ece(p[h].max(axis=1), ok), "pred": pred}
    jg = (gold.type.to_numpy() == "Incident") & (gold.priority.to_numpy() == "high")
    jp = (out["type"]["pred"] == "Incident") & (out["priority"]["pred"] == "high")
    out["joint"] = (int((jg & jp).sum()), int(jg.sum()), int(jp.sum()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["val300_flash_few8", "val300_lite_few8"])
    a = ap.parse_args()

    kb = pd.read_parquet(PROC / "kb_train.parquet")
    parts = pd.read_parquet(PROC / "kb_partitions.parquet")
    labels = load_labels(kb)
    queries, pool = select_rows("val", 300, kb, parts)
    gold = queries.reset_index(drop=True)
    tf = tfidf_probs(pool, queries, labels)

    lines = ["# Blend test: TF-IDF + Gemini on 300 val tickets", "",
             f"Same 300 `val` tickets as `val300_comparison.md`. TF-IDF fit on `fit`. Blend per head: "
             f"w x Gemini + (1 - w) x TF-IDF. **Pre-registered w = {PREREGISTERED}**; other rows are context only "
             "and must not be used to pick w (that happens on `calib`).", ""]
    for run in a.runs:
        lp, stated = llm_probs(run, gold.ticket_id, labels)
        lines += [f"## {run}", "",
                  "| w (Gemini share) | type F1 | queue F1 | priority F1 | type ECE | queue ECE | priority ECE | Incident+high recall | precision |",
                  "|---|---|---|---|---|---|---|---|---|"]
        res = {}
        for w in WEIGHTS:
            r = evaluate(gold, {h: w * lp[h] + (1 - w) * tf[h] for h in HEADS}, labels)
            res[w] = r
            tp, n, k = r["joint"]
            tag = " (pre-registered)" if w == PREREGISTERED else (" (TF-IDF only)" if w == 0 else (" (Gemini probs only)" if w == 1 else ""))
            lines.append(f"| {w}{tag} | " + " | ".join(f"{r[h]['macro_f1']:.3f}" for h in HEADS) + " | "
                         + " | ".join(f"{r[h]['ece']:.3f}" for h in HEADS)
                         + f" | {tp / n:.3f} ({tp}/{n}) | {tp / k if k else 0:.3f} ({tp}/{k}) |")
        stated_f1 = {h: f1_score(gold[h], stated[h], average="macro") for h in HEADS}
        lines += ["", "Gemini's stated label (not argmax of its probabilities): "
                  + ", ".join(f"{h} {stated_f1[h]:.3f}" for h in HEADS) + ".", ""]
        lines += ["Agreement between TF-IDF and Gemini argmax, and accuracy of the 50/50 blend on each part:", "",
                  "| head | agree | blend acc when agree | blend acc when disagree |", "|---|---|---|---|"]
        for h in HEADS:
            t_pred = np.array(labels[h])[tf[h].argmax(axis=1)]
            g_pred = np.array(labels[h])[lp[h].argmax(axis=1)]
            agree = t_pred == g_pred
            ok = res[PREREGISTERED][h]["pred"] == gold[h].to_numpy()
            lines.append(f"| {h} | {agree.mean():.3f} ({agree.sum()}/300) | {ok[agree].mean():.3f} | "
                         f"{ok[~agree].mean() if (~agree).any() else float('nan'):.3f} |")
        lines.append("")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "blend_val300.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
