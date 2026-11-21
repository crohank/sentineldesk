"""Refit only the priority head of the TF-IDF baseline on kb_train + extra_german rows.

    python scripts/experiment_german_priority.py A [B C ...]

Type and queue columns are copied from data/predictions/tfidf_logreg.parquet. Features and model match
scripts/baseline_tfidf.py, with the vectorizers fit on the variant's training text.
  A: critical -> high, very_low -> low, all German rows
  B: German rows labelled critical or very_low dropped
  C: mapping of A, German rows at sample_weight 0.5
  D: mapping of A, only rows whose queue_raw starts with "IT & Technology/"
  E: as D, critical and very_low dropped
  kb: KB only, nothing written; prints agreement with the baseline priority column as a reproduction check
Writes data/predictions/tfidf_prio_german_{A,B,C,D,E}.parquet. Score with scripts/run_eval.py.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
PRED = ROOT / "data" / "predictions"
BASE = PRED / "tfidf_logreg.parquet"
FIVE_TO_THREE = {"critical": "high", "high": "high", "medium": "medium", "low": "low", "very_low": "low"}
IT_PREFIX = "IT & Technology/"
VARIANTS = {"A": ("map", 1.0, False), "B": ("drop", 1.0, False), "C": ("map", 0.5, False),
            "D": ("map", 1.0, True), "E": ("drop", 1.0, True), "kb": (None, 0.0, False)}


def text(df: pd.DataFrame) -> pd.Series:
    return df["subject"] + " \n " + df["body"]


def training_set(variant: str, kb: pd.DataFrame, ger: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    how, weight, it_only = VARIANTS[variant]
    if how is None:
        return kb, np.ones(len(kb))
    if it_only:
        ger = ger[ger["queue_raw"].str.startswith(IT_PREFIX)]
    if how == "drop":
        ger = ger[ger["priority_raw"].isin(["high", "medium", "low"])]
    ger = ger.assign(priority=ger["priority_raw"].map(FIVE_TO_THREE))
    train = pd.concat([kb[["subject", "body", "priority"]], ger[["subject", "body", "priority"]]], ignore_index=True)
    return train, np.r_[np.ones(len(kb)), np.full(len(ger), weight)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="+", choices=list(VARIANTS))
    args = ap.parse_args()

    kb = pd.read_parquet(PROC / "kb_train.parquet", columns=["subject", "body", "priority"])
    ger = pd.read_parquet(PROC / "extra_german.parquet")
    ger = ger[~ger["holdout_twin"]]
    test = pd.read_parquet(PROC / "test_inputs.parquet")
    base = pd.read_parquet(BASE)
    assert (base["ticket_id"].to_numpy() == test["ticket_id"].to_numpy()).all()

    for v in args.variants:
        t0 = time.perf_counter()
        train, w = training_set(v, kb, ger)
        wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
        cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)
        Xtr = hstack([wv.fit_transform(text(train)), cv.fit_transform(text(train))]).tocsr()
        Xte = hstack([wv.transform(text(test)), cv.transform(text(test))]).tocsr()
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
        clf.fit(Xtr, train["priority"], sample_weight=w)
        proba = clf.predict_proba(Xte)
        pred = base.copy()
        pred["pred_priority"] = clf.classes_[proba.argmax(axis=1)]
        pred["conf_priority"] = proba.max(axis=1)
        print(f"{v}: {len(train)} training rows ({len(train) - len(kb)} German), "
              f"label mix {train['priority'].value_counts().to_dict()}, {time.perf_counter() - t0:.0f}s")
        print(f"  agreement with baseline priority: {(pred['pred_priority'] == base['pred_priority']).mean():.4f}, "
              f"predicted mix {pred['pred_priority'].value_counts().to_dict()}")
        if v == "kb":
            continue
        out = PRED / f"tfidf_prio_german_{v}.parquet"
        pred.to_parquet(out, index=False)
        print(f"  wrote {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
