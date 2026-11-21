"""Score a predictions file against the holdout labels.

The only script allowed to read test_labels.parquet. Format: docs/prediction_format.md.

    python scripts/score.py PREDICTIONS --run-name NAME [--exclude-ids FILE]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
TARGETS = ["type", "queue", "priority"]
REQUIRED = ["ticket_id"] + [f"pred_{t}" for t in TARGETS]
CONF_COLS = [f"conf_{t}" for t in TARGETS]
LIST_COLS = ["reason_codes", "retrieved_ids", "cited_ids"]
HUMAN_STATUSES = {"escalate_oncall", "human_review"}
ONCALL_STATUS = "escalate_oncall"
KNOWN_STATUSES = HUMAN_STATUSES | {"auto_draft", "needs_clarification"}
LANG_GROUPS = ["en", "de", "other"]
ECE_BINS = 10
RULE_NAMES = {
    "type_and_priority": "flag on pred Incident and pred high",
    "priority_only": "flag on pred high alone",
}


class ValidationError(Exception):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True, dtype={"ticket_id": str})
    else:
        raise ValidationError(f"unsupported extension {path.suffix!r}, use .parquet or .jsonl")
    for c in LIST_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda v: [] if v is None or (np.isscalar(v) and pd.isna(v)) else [str(x) for x in v])
    return df


def examples(values, k: int = 5) -> str:
    values = sorted(values)
    more = f" (+{len(values) - k} more)" if len(values) > k else ""
    return ", ".join(values[:k]) + more


def validate(pred: pd.DataFrame, holdout_ids: set[str], kb: pd.DataFrame) -> list[str]:
    errors = []
    missing_cols = [c for c in REQUIRED if c not in pred.columns]
    if missing_cols:
        raise ValidationError(f"missing required columns: {missing_cols}")
    for c in REQUIRED:
        n = int(pred[c].isna().sum())
        if n:
            errors.append(f"{c}: {n} null values")
    ids = pred["ticket_id"].astype(str)
    dup = ids[ids.duplicated()].unique()
    if len(dup):
        errors.append(f"repeated ticket_id: {len(dup)} IDs, e.g. {examples(dup)}")
    unknown = set(ids) - holdout_ids
    if unknown:
        errors.append(f"unknown ticket_id (not in holdout): {len(unknown)}, e.g. {examples(unknown)}")
    missing = holdout_ids - set(ids)
    if missing:
        errors.append(f"missing ticket_id: {len(missing)} of {len(holdout_ids)}, e.g. {examples(missing)}")
    for t in TARGETS:
        bad = set(pred[f"pred_{t}"].dropna().astype(str)) - set(kb[t])
        if bad:
            errors.append(f"pred_{t}: labels not in kb_train: {sorted(bad)}")
    for c in CONF_COLS:
        if c in pred.columns:
            v = pd.to_numeric(pred[c], errors="coerce")
            if v.isna().any() or ((v < 0) | (v > 1)).any():
                errors.append(f"{c}: values must be numbers in [0, 1]")
    kb_ids = set(kb["ticket_id"])
    for c in ["retrieved_ids", "cited_ids"]:
        if c in pred.columns:
            seen = {x for lst in pred[c] for x in lst}
            leaked = seen & holdout_ids
            if leaked:
                errors.append(f"{c}: holdout IDs used as KB precedents: {len(leaked)}, e.g. {examples(leaked)}")
            stray = seen - kb_ids - holdout_ids
            if stray:
                errors.append(f"{c}: IDs not in kb_train: {len(stray)}, e.g. {examples(stray)}")
    return errors


def rate(num: int, den: int) -> dict:
    return {"value": num / den if den else None, "num": int(num), "den": int(den)}


def fmt_rate(r: dict) -> str:
    v = "n/a" if r["value"] is None else f"{r['value']:.3f}"
    return f"{v} ({r['num']}/{r['den']})"


def classification_block(gold: pd.Series, pred: pd.Series, labels: list[str], majority: str) -> dict:
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=labels)
    per_class = {}
    for i, lab in enumerate(labels):
        tp = int(cm[i, i])
        per_class[lab] = {
            "precision": rate(tp, int(cm[:, i].sum())),
            "recall": rate(tp, int(s[i])),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
    maj_pred = pd.Series(majority, index=gold.index)
    return {
        "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": rate(int((gold == pred).sum()), len(gold)),
        "majority_label": majority,
        "majority_macro_f1": float(f1_score(gold, maj_pred, labels=labels, average="macro", zero_division=0)),
        "majority_accuracy": rate(int((gold == majority).sum()), len(gold)),
        "per_class": per_class,
        "confusion": {"labels": labels, "matrix": cm.tolist()},
    }


def outage_rules(df: pd.DataFrame) -> dict:
    gold = (df["type"] == "Incident") & (df["priority"] == "high")
    rules = {
        "type_and_priority": (df["pred_type"] == "Incident") & (df["pred_priority"] == "high"),
        "priority_only": df["pred_priority"] == "high",
    }
    out = {}
    for name, pred in rules.items():
        tp = int((gold & pred).sum())
        out[name] = {"flagged": int(pred.sum()), "precision": rate(tp, int(pred.sum())), "recall": rate(tp, int(gold.sum()))}
    return out


def outage_block(df: pd.DataFrame) -> dict:
    gold = (df["type"] == "Incident") & (df["priority"] == "high")
    out = {
        "definition": "type == Incident AND priority == high",
        "gold_events": int(gold.sum()),
        "rules": outage_rules(df),
    }
    if "decision_status" in df.columns:
        status = df["decision_status"].fillna("").astype(str)
        human = status.isin(HUMAN_STATUSES)
        oncall = status == ONCALL_STATUS
        out["routing"] = {
            "true_events_to_human": rate(int((gold & human).sum()), int(gold.sum())),
            "true_events_to_oncall": rate(int((gold & oncall).sum()), int(gold.sum())),
            "overall_escalation_rate": rate(int(human.sum()), len(df)),
            "overall_oncall_rate": rate(int(oncall.sum()), len(df)),
            "status_counts": {k: int(v) for k, v in status.value_counts().items()},
            "unrecognized_statuses": sorted(set(status) - KNOWN_STATUSES),
        }
        if "destination" in df.columns:
            dest = df.loc[gold, "destination"].fillna("(none)").astype(str)
            out["routing"]["true_event_destinations"] = {k: int(v) for k, v in dest.value_counts().items()}
    return out


def retrieval_block(df: pd.DataFrame, kb_labels: pd.DataFrame) -> dict:
    has = df["retrieved_ids"].map(len) > 0
    sub = df[has]
    out = {"rows_with_retrieval": rate(int(has.sum()), len(df))}
    for t in ["queue", "type"]:
        lookup = kb_labels[t]
        top1 = sub.apply(lambda r: lookup[r["retrieved_ids"][0]] == r[t], axis=1)
        top5 = sub.apply(lambda r: any(lookup[x] == r[t] for x in r["retrieved_ids"][:5]), axis=1)
        out[f"{t}_top1_agreement"] = rate(int(top1.sum()), len(sub))
        out[f"{t}_any_top5_agreement"] = rate(int(top5.sum()), len(sub))
    return out


def calibration_block(conf, correct, n_bins: int = ECE_BINS) -> dict:
    """Top-label ECE over equal-width bins (lo, hi]; a confidence of exactly 0 falls in the first bin."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1], right=True)
    bins, ece = [], 0.0
    for i in range(n_bins):
        m = idx == i
        n = int(m.sum())
        mean_conf = float(conf[m].mean()) if n else None
        acc = rate(int(correct[m].sum()), n)
        if n:
            ece += n / len(conf) * abs(acc["value"] - mean_conf)
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": n, "mean_conf": mean_conf,
                     "accuracy": acc, "gap": None if not n else acc["value"] - mean_conf})
    return {
        "ece": float(ece) if len(conf) else None,
        "n_bins": n_bins,
        "n": len(conf),
        "mean_conf": float(conf.mean()) if len(conf) else None,
        "accuracy": rate(int(correct.sum()), len(conf)),
        "bins": bins,
    }


def calibration_heads(df: pd.DataFrame) -> list[str]:
    return [t for t in TARGETS if f"conf_{t}" in df.columns]


def detect_language(body: str) -> str:
    try:
        code = detect(body) if body.strip() else "unknown"
    except LangDetectException:
        code = "unknown"
    return code if code in ("en", "de") else "other"


def language_breakdown(df: pd.DataFrame, col: str) -> dict:
    out = {}
    for lang in LANG_GROUPS:
        sub = df[df[col] == lang]
        if sub.empty:
            out[lang] = {"n": 0}
            continue
        # labels=None: a class absent from this slice in both gold and pred would otherwise score F1 0
        out[lang] = {
            "n": len(sub),
            "macro_f1": {t: float(f1_score(sub[t], sub[f"pred_{t}"], average="macro", zero_division=0)) for t in TARGETS},
            "accuracy": {t: rate(int((sub[t] == sub[f"pred_{t}"]).sum()), len(sub)) for t in TARGETS},
            "outage": outage_rules(sub),
        }
        heads = calibration_heads(sub)
        if heads:
            out[lang]["ece"] = {t: calibration_block(pd.to_numeric(sub[f"conf_{t}"]), sub[t] == sub[f"pred_{t}"])["ece"]
                                for t in heads}
    return out


def scope_metrics(df: pd.DataFrame, kb: pd.DataFrame, labels: dict, majority: dict) -> dict:
    m = {"n": len(df), "targets": {}}
    for t in TARGETS:
        m["targets"][t] = classification_block(df[t], df[f"pred_{t}"], labels[t], majority[t])
    m["outage"] = outage_block(df)
    heads = calibration_heads(df)
    if heads:
        m["calibration"] = {t: calibration_block(pd.to_numeric(df[f"conf_{t}"]), df[t] == df[f"pred_{t}"])
                            for t in heads}
    m["by_detected_language"] = language_breakdown(df, "detected_language")
    m["by_language_label"] = language_breakdown(df, "label_language")
    if "retrieved_ids" in df.columns:
        m["retrieval"] = retrieval_block(df, kb.set_index("ticket_id")[["queue", "type"]])
    if "cited_ids" in df.columns:
        m["citations"] = {"rows_with_citation": rate(int((df["cited_ids"].map(len) > 0).sum()), len(df))}
        if "retrieved_ids" in df.columns:
            cited = df["cited_ids"].map(len).sum()
            inside = sum(len(set(c) & set(r)) for c, r in zip(df["cited_ids"], df["retrieved_ids"]))
            m["citations"]["cited_ids_in_retrieved"] = rate(int(inside), int(cited))
    if "latency_ms" in df.columns:
        lat = pd.to_numeric(df["latency_ms"], errors="coerce").dropna()
        m["latency_ms"] = {"n": len(lat), "p50": float(lat.quantile(0.5)), "p95": float(lat.quantile(0.95)),
                           "max": float(lat.max())} if len(lat) else {"n": 0}
    return m


def plot_confusion(block: dict, title: str, path: Path) -> None:
    labels = block["confusion"]["labels"]
    cm = np.array(block["confusion"]["matrix"])
    row_share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    size = 3 + 0.6 * len(labels)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    ax.imshow(row_share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if row_share[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title} (counts, shaded by row share)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_reliability(block: dict, title: str, path: Path) -> None:
    bins = [b for b in block["bins"] if b["n"]]
    centers = [(b["lo"] + b["hi"]) / 2 for b in block["bins"]]
    width = 1 / block["n_bins"]
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(5, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], color="#9a9a9a", lw=1, ls="--", label="perfect calibration")
    ax.bar([(b["lo"] + b["hi"]) / 2 for b in bins], [b["accuracy"]["value"] for b in bins], width=width * 0.9,
           color="#6baed6", edgecolor="#2171b5", lw=0.8, label="accuracy in bin")
    ax.plot([b["mean_conf"] for b in bins], [b["accuracy"]["value"] for b in bins], "o-", color="#08306b",
            ms=4, lw=1.2, label="(mean confidence, accuracy)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}: ECE {block['ece']:.3f}, n {block['n']}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    axn.bar(centers, [b["n"] for b in block["bins"]], width=width * 0.9, color="#bdbdbd")
    for c, b in zip(centers, block["bins"]):
        if b["n"]:
            axn.text(c, b["n"], b["n"], ha="center", va="bottom", fontsize=6)
    axn.set_ylabel("tickets", fontsize=8)
    axn.set_xlabel("confidence in predicted label (10 equal-width bins)")
    axn.margins(y=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def headline_rows(m: dict) -> list[tuple[str, str]]:
    rows = []
    for t in TARGETS:
        b = m["targets"][t]
        rows.append((f"{t} macro-F1", f"{b['macro_f1']:.3f}"))
        rows.append((f"{t} accuracy", fmt_rate(b["accuracy"])))
    for name, desc in RULE_NAMES.items():
        r = m["outage"]["rules"][name]
        rows.append((f"Incident+high recall, {desc}", fmt_rate(r["recall"])))
        rows.append((f"Incident+high precision, {desc}", fmt_rate(r["precision"])))
    for t, c in m.get("calibration", {}).items():
        rows.append((f"{t} ECE, {c['n_bins']} bins", f"{c['ece']:.3f}"))
    if "routing" in m["outage"]:
        r = m["outage"]["routing"]
        rows.append(("true Incident+high routed to human", fmt_rate(r["true_events_to_human"])))
        rows.append(("true Incident+high routed to on-call", fmt_rate(r["true_events_to_oncall"])))
        rows.append(("overall escalation rate", fmt_rate(r["overall_escalation_rate"])))
        rows.append(("overall on-call rate", fmt_rate(r["overall_oncall_rate"])))
    if "retrieval" in m:
        for k, v in m["retrieval"].items():
            rows.append((f"retrieval {k.replace('_', ' ')}", fmt_rate(v)))
    return rows


def calibration_md(full: dict) -> list[str]:
    cal = full["calibration"]
    L = ["", "## Calibration (top-label confidence)", "",
         f"`conf_<head>` is the confidence in the predicted label. {ECE_BINS} equal-width bins, each (lo, hi]. "
         "ECE = sum over bins of (bin n / N) * |bin accuracy - bin mean confidence|. "
         "Gap = accuracy minus mean confidence, so negative means overconfident.", "",
         "| head | n | mean confidence | accuracy | ECE | plot |", "|---|---|---|---|---|---|"]
    for t, c in cal.items():
        L.append(f"| {t} | {c['n']} | {c['mean_conf']:.3f} | {fmt_rate(c['accuracy'])} | {c['ece']:.3f} "
                 f"| [reliability_{t}.png](reliability_{t}.png) |")
    for t, c in cal.items():
        L += ["", f"### Bins: {t}", "", "| bin | n | mean confidence | accuracy | gap |", "|---|---|---|---|---|"]
        for b in c["bins"]:
            if not b["n"]:
                L.append(f"| ({b['lo']:.1f}, {b['hi']:.1f}] | 0 | | | |")
                continue
            L.append(f"| ({b['lo']:.1f}, {b['hi']:.1f}] | {b['n']} | {b['mean_conf']:.3f} | {fmt_rate(b['accuracy'])} "
                     f"| {b['gap']:+.3f} |")
    heads = list(cal)
    L += ["", "### ECE by language detected from the body", "",
          "| language | n | " + " | ".join(f"{t} ECE" for t in heads) + " |", "|---|---|" + "---|" * len(heads)]
    for lang, g in full["by_detected_language"].items():
        cells = [f"{g['ece'][t]:.3f}" if g["n"] else "" for t in heads]
        L.append(f"| {lang} | {g['n']} | " + " | ".join(cells) + " |")
    return L


def render_md(meta: dict, full: dict, excluded: dict | None) -> str:
    L = [f"# Scores: {meta['run_name']}", ""]
    L += [f"- predictions: `{meta['predictions_path']}`",
          f"- sha256: `{meta['predictions_sha256']}`",
          f"- rows: {meta['rows']}",
          f"- scored at: {meta['scored_at']}",
          f"- git commit: {meta['git_commit']}"]
    if excluded is not None:
        L.append(f"- excluded IDs: {meta['excluded_ids']} from `{meta['exclude_ids_path']}`")
    L += ["", "Rates are shown as value (numerator/denominator).", "", "## Summary", ""]
    if excluded is None:
        L += ["| metric | full |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in headline_rows(full)]
    else:
        L += ["| metric | full | excluding listed IDs |", "|---|---|---|"]
        L += [f"| {k} | {v} | {w} |" for (k, v), (_, w) in zip(headline_rows(full), headline_rows(excluded))]

    L += ["", "## Against a training-majority predictor", "",
          "| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |",
          "|---|---|---|---|---|---|"]
    for t in TARGETS:
        b = full["targets"][t]
        L.append(f"| {t} | {b['majority_label']} | {b['majority_macro_f1']:.3f} | {fmt_rate(b['majority_accuracy'])} "
                 f"| {b['macro_f1']:.3f} | {fmt_rate(b['accuracy'])} |")

    for t in TARGETS:
        b = full["targets"][t]
        L += ["", f"## Per-class: {t}", "", "| class | support | precision | recall | F1 |", "|---|---|---|---|---|"]
        for lab, c in b["per_class"].items():
            L.append(f"| {lab} | {c['support']} | {fmt_rate(c['precision'])} | {fmt_rate(c['recall'])} | {c['f1']:.3f} |")
        L += ["", f"Confusion matrix: [confusion_{t}.png](confusion_{t}.png), [confusion_{t}.csv](confusion_{t}.csv)"]

    o = full["outage"]
    L += ["", "## Outage events (type == Incident AND priority == high)", "",
          f"True events: {o['gold_events']}. Two flagging rules:", "",
          "| rule | flagged | recall | precision |", "|---|---|---|---|"]
    for name, desc in RULE_NAMES.items():
        r = o["rules"][name]
        L.append(f"| {desc} | {r['flagged']} | {fmt_rate(r['recall'])} | {fmt_rate(r['precision'])} |")
    L.append("")
    if "routing" in o:
        r = o["routing"]
        L += [f"- true events routed to a human (escalate_oncall or human_review): {fmt_rate(r['true_events_to_human'])}",
              f"- true events routed to on-call: {fmt_rate(r['true_events_to_oncall'])}",
              f"- overall escalation rate (all tickets to a human): {fmt_rate(r['overall_escalation_rate'])}",
              f"- overall on-call rate: {fmt_rate(r['overall_oncall_rate'])}",
              f"- decision_status counts: {r['status_counts']}"]
        if r["unrecognized_statuses"]:
            L.append(f"- WARNING, unrecognized decision_status values (counted as not human): {r['unrecognized_statuses']}")
        if "true_event_destinations" in r:
            L.append(f"- destinations of true events: {r['true_event_destinations']}")
    else:
        L.append("- routing: no decision_status column")

    if "calibration" in full:
        L += calibration_md(full)

    for key, title, note in [
        ("by_detected_language", "By language detected from the body",
         "langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other."),
        ("by_language_label", "By language label", "The dataset's `language` column, which is wrong for about 12% of rows."),
    ]:
        L += ["", f"## {title}", "", note, "",
              "| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 "
              "| Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |",
              "|---|---|---|---|---|---|---|"]
        for lang, g in full[key].items():
            if not g["n"]:
                L.append(f"| {lang} | 0 | | | | | |")
                continue
            f = g["macro_f1"]
            L.append(f"| {lang} | {g['n']} | {f['type']:.3f} | {f['queue']:.3f} | {f['priority']:.3f} "
                     f"| {fmt_rate(g['outage']['type_and_priority']['recall'])} "
                     f"| {fmt_rate(g['outage']['priority_only']['recall'])} |")

    if "retrieval" in full:
        L += ["", "## Retrieval proxy (holdout label vs retrieved KB precedent labels)", ""]
        L += [f"- {k.replace('_', ' ')}: {fmt_rate(v)}" for k, v in full["retrieval"].items()]
    if "citations" in full:
        L += ["", "## Citations", ""]
        L += [f"- {k.replace('_', ' ')}: {fmt_rate(v)}" for k, v in full["citations"].items()]
    if "latency_ms" in full:
        L += ["", "## Latency (ms)", "", f"- {full['latency_ms']}"]
    if excluded is not None:
        L += ["", "Per-class numbers for the excluded subset are in metrics.json under `excluded`."]
    return "\n".join(L) + "\n"


def git_commit() -> str:
    """Short HEAD hash, with -dirty if tracked files have uncommitted changes (untracked files are ignored)."""
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        head = git("rev-parse", "--short", "HEAD")
        return head + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def shown(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


@lru_cache(maxsize=1)
def detected_languages() -> pd.DataFrame:
    DetectorFactory.seed = 0
    bodies = pd.read_parquet(PROC / "test_inputs.parquet", columns=["ticket_id", "body"])
    bodies["detected_language"] = bodies["body"].map(detect_language)
    return bodies[["ticket_id", "detected_language"]]


def score_run(predictions: Path, run_name: str, exclude_ids: Path | None = None) -> dict:
    """Score one predictions file into reports/RUN_NAME/ and return the metrics.json payload.

    Raises ValidationError (message starts with REJECTED) if the file breaks the format.
    """
    path = Path(predictions).resolve()
    digest = sha256(path)
    kb = pd.read_parquet(PROC / "kb_train.parquet", columns=["ticket_id"] + TARGETS)
    gold = pd.read_parquet(PROC / "test_labels.parquet", columns=["ticket_id", "language"] + TARGETS)
    gold = gold.rename(columns={"language": "label_language"})
    holdout_ids = set(gold["ticket_id"])

    try:
        pred = load_predictions(path)
        errors = validate(pred, holdout_ids, kb)
        exclude = None
        if exclude_ids:
            lines = Path(exclude_ids).read_text(encoding="utf-8").splitlines()
            exclude = {s.strip() for s in lines if s.strip() and not s.strip().startswith("#")}
            bad = exclude - holdout_ids
            if bad:
                errors.append(f"--exclude-ids: {len(bad)} IDs not in holdout, e.g. {examples(bad)}")
        if errors:
            raise ValidationError("\n".join(errors))
    except ValidationError as e:
        raise ValidationError(f"REJECTED {path.name} (sha256 {digest}):\n{e}") from None

    pred["ticket_id"] = pred["ticket_id"].astype(str)
    gold = gold.merge(detected_languages(), on="ticket_id", validate="one_to_one")
    df = gold.merge(pred, on="ticket_id", how="inner", validate="one_to_one")
    labels = {t: sorted(kb[t].unique()) for t in TARGETS}
    majority = {t: kb[t].value_counts().index[0] for t in TARGETS}

    full = scope_metrics(df, kb, labels, majority)
    excluded = scope_metrics(df[~df["ticket_id"].isin(exclude)], kb, labels, majority) if exclude else None

    meta = {
        "run_name": run_name,
        "predictions_path": shown(path),
        "predictions_sha256": digest,
        "rows": len(pred),
        "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "optional_columns": sorted(set(pred.columns) - set(REQUIRED)),
    }
    if exclude:
        meta["exclude_ids_path"] = shown(Path(exclude_ids))
        meta["excluded_ids"] = len(exclude)

    out = REPORTS / run_name
    out.mkdir(parents=True, exist_ok=True)
    for t in TARGETS:
        b = full["targets"][t]
        pd.DataFrame(b["confusion"]["matrix"], index=pd.Index(b["confusion"]["labels"], name="true"),
                     columns=b["confusion"]["labels"]).to_csv(out / f"confusion_{t}.csv")
        plot_confusion(b, f"{run_name}: {t}", out / f"confusion_{t}.png")
    for t, c in full.get("calibration", {}).items():
        plot_reliability(c, f"{run_name}: {t}", out / f"reliability_{t}.png")
    payload = {"meta": meta, "full": full}
    if excluded is not None:
        payload["excluded"] = excluded
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.md").write_text(render_md(meta, full, excluded), encoding="utf-8")
    return payload


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--exclude-ids", type=Path, help="text file, one holdout ticket_id per line")
    args = ap.parse_args()
    try:
        payload = score_run(args.predictions, args.run_name, args.exclude_ids)
    except ValidationError as e:
        print(e, file=sys.stderr)
        return 2
    for k, v in headline_rows(payload["full"]):
        print(f"{k}: {v}")
    print(f"wrote {(REPORTS / args.run_name).relative_to(ROOT).as_posix()}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
