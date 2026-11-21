"""Clean the German CSV for use as extra priority training data.

Reads data/raw/dataset-tickets-german_normalized_50_5_2.csv, kb_train and test_inputs (never test_labels).
Cleans with the make_split.py rules, drops empty bodies, repeats inside the file and bodies that match a
KB or holdout ticket, then drops holdout paraphrase twins (word 1-2 gram TF-IDF cosine >= 0.8, the
twin_audit.py rule). Runs langdetect on a sample. Writes data/processed/extra_german.parquet with IDs
GER-00000.. and prints the counts for reports/audit_german.md.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "dataset-tickets-german_normalized_50_5_2.csv"
PROC = ROOT / "data" / "processed"
OUT = PROC / "extra_german.parquet"
TWIN = 0.8
LANG_SAMPLE = 2000


def key(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower()


def ticket(d: pd.DataFrame) -> pd.Series:
    return d["subject"] + " \n " + d["body"]


def max_cosine(fit_text: pd.Series, a: pd.Series, b: pd.Series) -> np.ndarray:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(fit_text)
    A, B = vec.transform(a), vec.transform(b)
    return np.concatenate([(A[s:s + 1000] @ B.T).max(axis=1).toarray().ravel()
                           for s in range(0, A.shape[0], 1000)])


def lang(body: str) -> str:
    try:
        return detect(body)
    except LangDetectException:
        return "unknown"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    counts = {"sha256": hashlib.sha256(RAW.read_bytes()).hexdigest()}
    raw = pd.read_csv(RAW)
    counts["raw"] = len(raw)
    x = raw.copy()
    for c in ["subject", "body"]:
        x[c] = x[c].fillna("").str.replace("\\n", "\n", regex=False)
    x = x[x["body"].str.strip() != ""]
    counts["empty body"] = counts["raw"] - len(x)

    x["k"] = key(x["body"])
    n = len(x)
    x = x.drop_duplicates("k", keep="first")
    counts["repeated inside file"] = n - len(x)

    kb = pd.read_parquet(PROC / "kb_train.parquet", columns=["subject", "body"])
    te = pd.read_parquet(PROC / "test_inputs.parquet", columns=["ticket_id", "subject", "body"])
    in_kb, in_te = x["k"].isin(set(key(kb["body"]))), x["k"].isin(set(key(te["body"])))
    counts["matches KB body"] = int(in_kb.sum())
    counts["matches holdout body"] = int(in_te.sum())
    x = x[~in_kb & ~in_te].reset_index(drop=True)
    counts["after exact-match drops"] = len(x)

    sim = max_cosine(pd.concat([ticket(kb), ticket(x)]), ticket(x), ticket(te))
    x["holdout_sim"] = sim
    x["holdout_twin"] = sim >= TWIN
    counts["holdout twins"] = int(x["holdout_twin"].sum())
    counts["holdout sim bands"] = {str(b): int((sim >= b).sum()) for b in [0.5, 0.6, 0.7, 0.8, 0.9]}
    x = x[~x["holdout_twin"]].reset_index(drop=True)
    counts["kept"] = len(x)

    x["ticket_id"] = [f"GER-{i:05d}" for i in range(len(x))]
    x = x.rename(columns={"priority": "priority_raw"})
    x[["ticket_id", "subject", "body", "priority_raw", "holdout_twin", "holdout_sim"]].to_parquet(OUT, index=False)
    counts["priority_raw"] = x["priority_raw"].value_counts().to_dict()

    DetectorFactory.seed = 0
    sample = x.sample(n=min(LANG_SAMPLE, len(x)), random_state=42)
    detected = sample["body"].map(lang).value_counts()
    counts["langdetect sample"] = {"n": len(sample), **detected.to_dict()}
    counts["language column"] = raw["language"].value_counts().to_dict()

    print(json.dumps(counts, indent=1, ensure_ascii=False))
    print(f"wrote {OUT.relative_to(ROOT).as_posix()} ({len(x)} rows)")


if __name__ == "__main__":
    main()
