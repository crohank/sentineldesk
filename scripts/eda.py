"""Exploratory stats on the KB split. Writes docs/eda_stats.txt and docs/figures/*.png.

Reads test_labels only for the train/test distribution check and baseline scores;
never reads holdout answers (they are not on disk).
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from langdetect import DetectorFactory, detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
TARGETS = ["type", "queue", "priority"]

out_lines: list[str] = []


def emit(*parts) -> None:
    line = " ".join(str(p) for p in parts)
    out_lines.append(line)
    print(line)


def section(title: str) -> None:
    emit("")
    emit("=" * 8, title, "=" * 8)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(parents=True, exist_ok=True)
    kb = pd.read_parquet(PROC / "kb_train.parquet")
    test_x = pd.read_parquet(PROC / "test_inputs.parquet")
    test_y = pd.read_parquet(PROC / "test_labels.parquet")
    test = test_x.merge(test_y, on="ticket_id")

    section("Sizes")
    emit("kb", kb.shape, "test", test.shape)
    emit("version column values:", kb["version"].value_counts().to_dict())

    section("Label distributions (kb, %)")
    for c in TARGETS + ["language"]:
        emit(f"-- {c}")
        emit((kb[c].value_counts(normalize=True) * 100).round(1).to_string())

    section("Train vs test label share drift (max abs pct-point gap)")
    for c in TARGETS + ["language"]:
        a = kb[c].value_counts(normalize=True)
        b = test[c].value_counts(normalize=True)
        emit(c, round(float((a - b).abs().max() * 100), 2))

    section("Escalation rule base rate: priority==high & type==Incident")
    hi = (kb["priority"] == "high") & (kb["type"] == "Incident")
    emit(f"{hi.sum()} rows, {hi.mean() * 100:.1f}% of kb")
    emit("type x priority (row %):")
    emit((pd.crosstab(kb["type"], kb["priority"], normalize="index") * 100).round(1).to_string())
    emit("queue x priority (row %):")
    emit((pd.crosstab(kb["queue"], kb["priority"], normalize="index") * 100).round(1).to_string())
    emit("queue x type (row %):")
    emit((pd.crosstab(kb["queue"], kb["type"], normalize="index") * 100).round(1).to_string())

    section("Labels by language (row %)")
    for c in TARGETS:
        emit(f"-- {c}")
        emit((pd.crosstab(kb["language"], kb[c], normalize="index") * 100).round(1).to_string())

    section("Missing subjects")
    empty = kb["subject"].str.strip() == ""
    emit(f"empty subject: {empty.sum()} ({empty.mean() * 100:.1f}%)")
    emit("by language:", kb[empty]["language"].value_counts().to_dict())
    emit("empty answers in kb:", int((kb["answer"].str.strip() == "").sum()))

    section("Text lengths (words)")
    for col in ["subject", "body", "answer"]:
        w = kb[col].str.split().str.len()
        emit(col, w.describe(percentiles=[0.5, 0.9, 0.99]).round(0).to_dict())
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for col in ["body", "answer"]:
        kb[col].str.split().str.len().clip(upper=400).plot.hist(bins=60, alpha=0.6, ax=ax, label=col)
    ax.set_xlabel("words (clipped at 400)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "length_hist.png", dpi=110)

    section("Tags")
    tag_cols = [f"tag_{i}" for i in range(1, 9)]
    tags = kb[tag_cols].stack()
    emit("distinct tags:", tags.nunique(), " tags/row mean:", round(kb[tag_cols].notna().sum(axis=1).mean(), 2))
    emit(tags.value_counts().head(30).to_string())

    section("Language column vs langdetect (500-row sample)")
    DetectorFactory.seed = 0
    s = kb.sample(500, random_state=0)
    det = s["body"].map(lambda t: detect(t) if t.strip() else "??")
    emit(f"agreement: {(det == s['language']).mean() * 100:.1f}%")
    emit("disagreements:", s.loc[det != s["language"], "language"].value_counts().to_dict(),
         "detected as:", det[det != s["language"]].value_counts().to_dict())

    section("PII-like patterns in body (kb)")
    pats = {
        "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
        "ipv4": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
        "phone": r"\+?\d[\d\s().-]{8,}\d",
        "placeholder <...>": r"<[^<>]{2,30}>",
        "placeholder [...]": r"\[[^\[\]]{2,30}\]",
    }
    for name, p in pats.items():
        hit = kb["body"].str.contains(p, regex=True)
        emit(f"{name}: {hit.mean() * 100:.1f}%  e.g. {kb.loc[hit, 'body'].str.extract('(' + p + ')')[0].head(5).tolist()}")
    emit("answer placeholders [...]:", kb["answer"].str.extractall(r"(\[[^\[\]]{2,30}\])")[0].value_counts().head(10).to_dict())
    emit("answer placeholders <...>:", kb["answer"].str.extractall(r"(<[^<>]{2,30}>)")[0].value_counts().head(10).to_dict())

    section("Answer content risks (numbers / money / guarantees)")
    for name, p in {
        "money": r"[$€£]\s?\d|\d\s?(?:USD|EUR|euro)",
        "percent": r"\d+\s?%",
        "refund": r"refund|rückerstattung|erstatt",
        "guarantee": r"guarantee|garantie",
        "SLA/hours": r"\b\d+\s?(?:hours|stunden|days|tage)\b",
    }.items():
        emit(f"{name}: {kb['answer'].str.contains(p, case=False, regex=True).mean() * 100:.1f}% of answers")

    section("Sample tickets")
    for _, r in kb.sample(4, random_state=1).iterrows():
        emit(f"[{r.ticket_id}] {r.language} | {r.type} | {r.queue} | {r.priority}")
        emit("  SUBJ:", r.subject[:120])
        emit("  BODY:", r.body[:300].replace("\n", " "))
        emit("  ANSW:", r.answer[:300].replace("\n", " "))

    section("Baseline: TF-IDF (word 1-2 + char 3-5) + LogisticRegression")
    text_kb = (kb["subject"] + " \n " + kb["body"])
    text_te = (test["subject"] + " \n " + test["body"])
    from scipy.sparse import hstack
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([wv.fit_transform(text_kb), cv.fit_transform(text_kb)]).tocsr()
    Xte = hstack([wv.transform(text_te), cv.transform(text_te)]).tocsr()
    preds = {}
    for c in TARGETS:
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
        clf.fit(Xtr, kb[c])
        preds[c] = clf.predict(Xte)
        emit(f"{c}: macro-F1={f1_score(test[c], preds[c], average='macro'):.3f}  "
             f"acc={(preds[c] == test[c]).mean():.3f}  majority-acc={test[c].value_counts(normalize=True).iloc[0]:.3f}")
    gold_hi = (test["priority"] == "high") & (test["type"] == "Incident")
    pred_hi = (preds["priority"] == "high") & (preds["type"] == "Incident")
    emit(f"high-priority Incident recall: {recall_score(gold_hi, pred_hi):.3f} (n={gold_hi.sum()})")
    emit(f"priority==high recall: {recall_score(test['priority'] == 'high', preds['priority'] == 'high'):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, c in zip(axes, TARGETS):
        kb[c].value_counts().sort_values().plot.barh(ax=ax)
        ax.set_title(c)
    fig.tight_layout()
    fig.savefig(FIG / "label_dists.png", dpi=110)

    (DOCS / "eda_stats.txt").write_text("\n".join(out_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
