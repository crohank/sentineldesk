"""TF-IDF (word 1-2 + char_wb 3-5) + LogisticRegression baseline, one classifier per head.

Fits on kb_train, predicts test_inputs. Reads nothing else.
Writes data/predictions/tfidf_logreg.parquet with conf_* = max class probability.
`--extra` adds the non-twin rows of data/processed/extra_20k.parquet (scripts/audit_20k.py)
to training only.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "predictions" / "tfidf_logreg.parquet"
TARGETS = ["type", "queue", "priority"]


def text(df: pd.DataFrame) -> pd.Series:
    return df["subject"] + " \n " + df["body"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--extra", action="store_true", help="also train on extra_20k rows that are not holdout twins")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out.resolve()

    kb = pd.read_parquet(PROC / "kb_train.parquet")
    if args.extra:
        extra = pd.read_parquet(PROC / "extra_20k.parquet")
        extra = extra[~extra["holdout_twin"]]
        kb = pd.concat([kb, extra[kb.columns.intersection(extra.columns)]], ignore_index=True)
        print(f"training on kb_train + {len(extra)} extra_20k rows = {len(kb)}")
    test = pd.read_parquet(PROC / "test_inputs.parquet")
    t0 = time.perf_counter()
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([wv.fit_transform(text(kb)), cv.fit_transform(text(kb))]).tocsr()
    Xte = hstack([wv.transform(text(test)), cv.transform(text(test))]).tocsr()

    pred = test[["ticket_id"]].copy()
    for t in TARGETS:
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
        clf.fit(Xtr, kb[t])
        proba = clf.predict_proba(Xte)
        pred[f"pred_{t}"] = clf.classes_[proba.argmax(axis=1)]
        pred[f"conf_{t}"] = proba.max(axis=1)
        print(f"{t}: fitted ({time.perf_counter() - t0:.0f}s)")
    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out, index=False)
    print(f"wrote {out.relative_to(ROOT).as_posix()} ({len(pred)} rows)")


if __name__ == "__main__":
    main()
