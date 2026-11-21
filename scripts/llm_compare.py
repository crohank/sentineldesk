"""Like-for-like comparison of classifiers on the 300 val tickets that llm_classify.py scores.

Rows: fit-partition majority class, TF-IDF + LogReg fit on the `fit` partition only, a vote over
the same k=8 fit neighbours the LLM saw as few-shot examples, and every LLM run found as
reports/llm/val300_*.json (or the names given with --runs). Every row is scored on the same
ticket IDs; a run with a different ID set stops the script.

Reads kb_train, kb_partitions and the val prediction files. Never reads test_labels.parquet.
Writes reports/llm/val300_comparison.md and .json.

    python scripts/llm_compare.py
    python scripts/llm_compare.py --runs val300_flash_few8 val300_lite_few8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import DetectorFactory
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import llm_classify as lc  # noqa: E402
from scripts.gemini_client import PRICES  # noqa: E402
from scripts.score import calibration_block, detect_language, fmt_rate, outage_rules, rate  # noqa: E402
from scripts.twin_audit import max_kb_similarity  # noqa: E402

ROOT = lc.ROOT
REPORTS = lc.REPORTS
PRED_VAL = lc.PRED_DIR / "val"
HEADS = lc.HEADS
OUT_NAME = "val300_comparison"
N_QUERIES = 300
K = 8
TWIN_THRESHOLD = 0.8
N_BOOT = 1000
SEED = 42
REFERENCE = "tfidf_logreg_fit"
# reports/llm_relabel_pilot.md: gemini-3.8-flash, prompt relabel-v1, 10 fixed German examples (one per
# queue) for every ticket. Copied because that run stores no per-ticket predictions.
PILOT_LLM = {"type": 0.705, "queue": 0.220, "priority": 0.408}
JOINT_RULES = {"type_and_priority": "pred Incident and pred high", "priority_only": "pred high alone"}


class CompareError(RuntimeError):
    pass


def assert_same_ids(rows: dict[str, pd.DataFrame], expected) -> None:
    expected = set(expected)
    for name, df in rows.items():
        ids = df["ticket_id"]
        if ids.duplicated().any():
            raise CompareError(f"{name}: duplicate ticket IDs, e.g. {ids[ids.duplicated()].iloc[0]}")
        got = set(ids)
        if got != expected:
            missing, extra = sorted(expected - got), sorted(got - expected)
            raise CompareError(f"{name}: ticket IDs differ from the val300 selection "
                               f"({len(missing)} missing, e.g. {missing[:3]}; {len(extra)} extra, e.g. {extra[:3]})")


def _macro_f1_codes(g: np.ndarray, p: np.ndarray, n_labels: int) -> float:
    # Averages over labels present in gold or pred, which is sklearn's f1_score default and what
    # llm_classify.py reports, so the headline numbers match the run summaries.
    cm = np.bincount(g * n_labels + p, minlength=n_labels * n_labels).reshape(n_labels, n_labels)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    present = denom > 0
    return float((2 * np.diag(cm)[present] / denom[present]).mean())


def _encode(*series) -> tuple[list[np.ndarray], int]:
    vocab = sorted(set().union(*(set(s) for s in series)))
    lookup = {v: i for i, v in enumerate(vocab)}
    return [np.array([lookup[v] for v in s], dtype=np.int64) for s in series], len(vocab)


def macro_f1(gold, pred) -> float:
    (g, p), n = _encode(list(gold), list(pred))
    return _macro_f1_codes(g, p, n)


def paired_bootstrap(gold, pred_a, pred_b, *, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """95% percentile CI of macro-F1(a) minus macro-F1(b), resampling tickets with replacement."""
    (g, a, b), n_labels = _encode(list(gold), list(pred_a), list(pred_b))
    n = len(g)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    diffs = np.array([_macro_f1_codes(g[i], a[i], n_labels) - _macro_f1_codes(g[i], b[i], n_labels) for i in idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": _macro_f1_codes(g, a, n_labels) - _macro_f1_codes(g, b, n_labels), "lo": float(lo),
            "hi": float(hi), "share_le_0": float((diffs <= 0).mean()), "n": n, "n_boot": n_boot, "seed": seed,
            "excludes_0": bool(lo > 0 or hi < 0)}


def twin_flags(pool: pd.DataFrame, queries: pd.DataFrame, threshold: float = TWIN_THRESHOLD) -> pd.DataFrame:
    """twin_audit.py's method with the fit partition standing in for the KB."""
    best, arg = max_kb_similarity(pool["subject"] + " \n " + pool["body"], queries["subject"] + " \n " + queries["body"])
    return pd.DataFrame({"ticket_id": queries["ticket_id"].to_numpy(), "twin_sim": best,
                         "twin_id": pool["ticket_id"].to_numpy()[arg], "is_twin": best >= threshold})


def vote(labels: list[str]) -> tuple[str, float]:
    """Most common label; a tie goes to the label of the most similar neighbour among the tied ones."""
    counts = Counter(labels)
    top = max(counts.values())
    winner = next(l for l in labels if counts[l] == top)
    return winner, top / len(labels)


def knn_predictions(pool: pd.DataFrame, queries: pd.DataFrame, nbrs: list[list[int]]) -> pd.DataFrame:
    out = {"ticket_id": queries["ticket_id"].to_numpy()}
    for h in HEADS:
        votes = [vote(pool[h].iloc[row].tolist()) for row in nbrs]
        out[f"pred_{h}"] = [v[0] for v in votes]
        out[f"conf_{h}"] = [v[1] for v in votes]
    return pd.DataFrame(out)


def majority_predictions(pool: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"ticket_id": queries["ticket_id"].to_numpy()})
    for h in HEADS:
        out[f"pred_{h}"] = pool[h].mode().iloc[0]
    return out


def tfidf_fit(pool: pd.DataFrame):
    """baseline_tfidf.py's recipe, fit on the fit partition instead of the whole KB. Returns (predict, fit seconds)."""
    def text(df):
        return df["subject"] + " \n " + df["body"]
    t0 = time.perf_counter()
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)
    Xtr = hstack([wv.fit_transform(text(pool)), cv.fit_transform(text(pool))]).tocsr()
    models = {h: LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(Xtr, pool[h]) for h in HEADS}

    def predict(queries: pd.DataFrame) -> pd.DataFrame:
        Xq = hstack([wv.transform(text(queries)), cv.transform(text(queries))]).tocsr()
        out = pd.DataFrame({"ticket_id": queries["ticket_id"].to_numpy()})
        for h, clf in models.items():
            proba = clf.predict_proba(Xq)
            out[f"pred_{h}"] = clf.classes_[proba.argmax(axis=1)]
            out[f"conf_{h}"] = proba.max(axis=1)
        return out
    return predict, time.perf_counter() - t0


def german_pilot_check(tagged: pd.DataFrame, pool: pd.DataFrame, tfidf_predict) -> dict:
    """Local rows on the ticket set of reports/llm_relabel_pilot.md, to separate ticket set from prompt.

    The selection mirrors scripts/llm_relabel_pilot.py:121-125 (val partition, langdetect de with
    seed 0, 300 sampled with random_state 42).
    """
    DetectorFactory.seed = 0
    val = tagged[tagged["partition"] == "val"]
    de = val[val["body"].map(detect_language) == "de"]
    pilot = de.sample(n=min(N_QUERIES, len(de)), random_state=SEED).sort_values("ticket_id").reset_index(drop=True)
    knn = knn_predictions(pool, pilot, lc.neighbours(pool, pilot, K))
    tf = tfidf_predict(pilot)
    return {"n": len(pilot), "val_german": len(de),
            "knn_vote_k8": {h: macro_f1(pilot[h], knn[f"pred_{h}"]) for h in HEADS},
            REFERENCE: {h: macro_f1(pilot[h], tf[f"pred_{h}"]) for h in HEADS},
            "pilot_llm_reported": PILOT_LLM}


def find_runs(names: list[str] | None) -> list[str]:
    if names:
        return names
    return sorted(p.stem for p in REPORTS.glob("val300_*.json") if p.stem != OUT_NAME)


def load_run(name: str) -> tuple[dict, pd.DataFrame]:
    summary = json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8"))
    meta = summary["meta"]
    if meta.get("split") != "val":
        raise CompareError(f"{name}: split is {meta.get('split')!r}, not val")
    pred = pd.read_parquet(ROOT / meta["predictions"])
    return summary, pred


def head_metrics(df: pd.DataFrame, h: str) -> dict:
    return {"macro_f1": macro_f1(df[h], df[f"pred_{h}"]),
            "accuracy": rate(int((df[h] == df[f"pred_{h}"]).sum()), len(df))}


def subset_f1(df: pd.DataFrame, mask) -> dict:
    sub = df[mask]
    return {"n": len(sub), "macro_f1": {h: macro_f1(sub[h], sub[f"pred_{h}"]) if len(sub) else None for h in HEADS}}


def row_metrics(df: pd.DataFrame) -> dict:
    """df: gold type/queue/priority, pred_*, optional conf_*, detected_language, is_twin."""
    m = {"heads": {h: head_metrics(df, h) for h in HEADS}, "joint_incident_high": outage_rules(df)}
    m["ece"] = {h: calibration_block(df[f"conf_{h}"].astype(float), df[h] == df[f"pred_{h}"])["ece"]
                for h in HEADS if f"conf_{h}" in df.columns}
    m["language"] = {lang: subset_f1(df, df["detected_language"] == lang) for lang in ("en", "de")}
    m["twins"] = {"twin": subset_f1(df, df["is_twin"]), "non_twin": subset_f1(df, ~df["is_twin"])}
    return m


def compare(run_names: list[str] | None = None, *, with_pilot: bool = True) -> dict:
    kb = pd.read_parquet(lc.PROC / "kb_train.parquet")
    parts = pd.read_parquet(lc.PROC / "kb_partitions.parquet")
    queries, pool = lc.select_rows("val", N_QUERIES, kb, parts)
    ids = queries["ticket_id"].tolist()

    DetectorFactory.seed = 0
    gold = queries[["ticket_id", "body", *HEADS]].assign(detected_language=queries["body"].map(detect_language))
    twins = twin_flags(pool, queries)
    gold = gold.merge(twins, on="ticket_id", validate="one_to_one").drop(columns="body")

    t0 = time.perf_counter()
    nbrs = lc.neighbours(pool, queries, K)
    knn_ms = (time.perf_counter() - t0) * 1000 / len(queries)
    nbr_ids = {t: [pool["ticket_id"].iat[j] for j in row] for t, row in zip(queries["ticket_id"], nbrs)}

    preds: dict[str, pd.DataFrame] = {}
    info: dict[str, dict] = {}
    preds["majority_fit"] = majority_predictions(pool, queries)
    info["majority_fit"] = {"label": "Majority class (fit)", "kind": "local", "cost_usd": 0.0,
                            "latency_ms_p50": 0.0, "latency_ms_p95": 0.0,
                            "labels": {h: preds["majority_fit"][f"pred_{h}"].iat[0] for h in HEADS}}
    tfidf_predict, fit_s = tfidf_fit(pool)
    t0 = time.perf_counter()
    preds[REFERENCE] = tfidf_predict(queries)
    timing = {"fit_seconds": round(fit_s, 1),
              "predict_ms_per_ticket_batch": round((time.perf_counter() - t0) * 1000 / len(queries), 3)}
    info[REFERENCE] = {"label": "TF-IDF + LogReg (fit)", "kind": "local", "cost_usd": 0.0,
                       "latency_ms_p50": timing["predict_ms_per_ticket_batch"],
                       "latency_ms_p95": timing["predict_ms_per_ticket_batch"], **timing}
    preds["knn_vote_k8"] = knn_predictions(pool, queries, nbrs)
    info["knn_vote_k8"] = {"label": f"kNN vote (k={K}, LLM's few-shot set)", "kind": "local", "cost_usd": 0.0,
                           "latency_ms_p50": round(knn_ms, 3), "latency_ms_p95": round(knn_ms, 3)}

    for name in find_runs(run_names):
        summary, pred = load_run(name)
        meta = summary["meta"]
        assert_same_ids({name: pred}, ids)
        mismatched = [t for t, f in zip(pred["ticket_id"], pred["fewshot_ids"]) if list(f) != nbr_ids[t]]
        if mismatched:
            raise CompareError(f"{name}: few-shot IDs differ from the rebuilt k={K} neighbours on "
                               f"{len(mismatched)} tickets, e.g. {mismatched[:3]}")
        u = meta.get("usage", {})
        calls = u.get("calls", 0)
        tokens = u.get("prompt_tokens", 0) + u.get("output_tokens", 0) + u.get("thought_tokens", 0)
        info[name] = {
            "label": f"{meta['model']} ({meta.get('thinking_level') or 'default'})", "kind": "llm",
            "model": meta["model"], "thinking_level": meta.get("thinking_level"), "backend": meta.get("backend"),
            "mode": meta.get("mode"), "k": meta.get("k"), "prompt_version": meta.get("prompt_version"),
            "few_shot_pool": meta.get("few_shot_pool"), "generated_utc": meta.get("generated_utc"),
            "cost_usd": meta.get("est_cost_usd", 0.0), "calls": calls, "cached": u.get("cached", 0),
            "tokens_per_call": {"prompt": u.get("prompt_tokens", 0) / calls if calls else None,
                                "output": u.get("output_tokens", 0) / calls if calls else None,
                                "thought": u.get("thought_tokens", 0) / calls if calls else None,
                                "total": tokens / calls if calls else None},
            "cost_per_1k_tickets_usd": meta.get("est_cost_usd", 0.0) / calls * 1000 if calls else None,
            "price_per_1m_in_out_usd": PRICES.get(meta["model"]),
            "latency_ms_p50": float(pred["latency_ms"].quantile(0.5)),
            "latency_ms_p95": float(pred["latency_ms"].quantile(0.95)),
            "stored_macro_f1": {h: summary.get("macro_f1", {}).get(h) for h in HEADS},
        }
        preds[name] = pred

    assert_same_ids(preds, ids)
    rows = {}
    for name, pred in preds.items():
        cols = ["ticket_id"] + [c for c in pred.columns if c.startswith(("pred_", "conf_"))]
        df = gold.merge(pred[cols], on="ticket_id", validate="one_to_one")
        rows[name] = {**info[name], "metrics": row_metrics(df), "_df": df}
        stored = info[name].get("stored_macro_f1")
        if stored:
            for h in HEADS:
                if stored[h] is not None and abs(stored[h] - rows[name]["metrics"]["heads"][h]["macro_f1"]) > 1e-3:
                    raise CompareError(f"{name}: recomputed {h} macro-F1 does not match the run summary")

    ref = rows[REFERENCE]["_df"]
    non_twin = ~ref["is_twin"].to_numpy()
    for name, r in rows.items():
        if name == REFERENCE:
            continue
        df = r["_df"]
        r["bootstrap_vs_tfidf"] = {h: paired_bootstrap(df[h], df[f"pred_{h}"], ref[f"pred_{h}"]) for h in HEADS}
        r["bootstrap_vs_tfidf_non_twin"] = {
            h: paired_bootstrap(df[h][non_twin], df[f"pred_{h}"][non_twin], ref[f"pred_{h}"][non_twin]) for h in HEADS}

    tw = gold[gold["is_twin"]]
    twin_lookup = pool.set_index("ticket_id")
    copy = {h: rate(int((twin_lookup.loc[tw["twin_id"], h].to_numpy() == tw[h].to_numpy()).sum()), len(tw))
            for h in HEADS}
    twin_in_fewshot = rate(sum(t in nbr_ids[q] for q, t in zip(tw["ticket_id"], tw["twin_id"])), len(tw))

    for r in rows.values():
        r.pop("_df")
    pilot = german_pilot_check(kb.merge(parts, on="ticket_id"), pool, tfidf_predict) if with_pilot else None
    return {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n": len(ids), "ticket_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
            "selection": "llm_classify.select_rows('val', 300): stratified on type|priority, seed 42",
            "pool": f"fit partition ({len(pool)} tickets)", "k": K, "reference_row": REFERENCE,
            "bootstrap": {"n_boot": N_BOOT, "seed": SEED, "ci": 0.95},
            "twin_threshold": TWIN_THRESHOLD, "ece_bins": 10,
        },
        "population": {
            "languages": {k: int(v) for k, v in gold["detected_language"].value_counts().items()},
            "twins": int(gold["is_twin"].sum()),
            "twin_ids": sorted(tw["ticket_id"]),
            "twin_label_copy_accuracy": copy,
            "twin_in_fewshot": twin_in_fewshot,
            "gold_incident_high": int(((gold["type"] == "Incident") & (gold["priority"] == "high")).sum()),
        },
        "rows": rows,
        "german_pilot_check": pilot,
    }


def f3(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def ms(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.0f}" if x >= 10 else f"{x:.2f}"


def ci(b: dict) -> str:
    return f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]{' *' if b['excludes_0'] else ''}"


def triple(d: dict) -> str:
    return " / ".join(f3(d[h]) for h in HEADS)


def german_md(res: dict) -> list[str]:
    rows, chk = res["rows"], res.get("german_pilot_check")
    de = {name: r["metrics"]["language"]["de"] for name, r in rows.items()}
    L = ["", "### German only", "",
         "| row | n | type / queue / priority |", "|---|---|---|",
         *[f"| {rows[k]['label']} | {s['n']} | {triple(s['macro_f1'])} |" for k, s in de.items()]]
    if not chk:
        return L
    tf_here, tf_there = de[REFERENCE]["macro_f1"], chk[REFERENCE]
    set_shift = max(abs(tf_here[h] - tf_there[h]) for h in HEADS)
    below_vote = all(chk["pilot_llm_reported"][h] < chk["knn_vote_k8"][h] for h in HEADS)
    verdict = ("so most of the gap is the prompt (10 fixed examples, one per queue, for every ticket, instead of "
               "the 8 retrieved neighbours): neither the ticket set nor the language explains an LLM that scores "
               "below a plain neighbour vote on its own tickets." if below_vote else
               "so the prompt alone does not explain the gap; see the per-head numbers.")
    L += ["",
          f"Cross-check with `reports/llm_relabel_pilot.md`, where gemini-3.8-flash with its own prompt scored "
          f"{triple(chk['pilot_llm_reported'])} on {chk['n']} German val tickets (of {chk['val_german']}). On "
          f"that reproduced ticket set, TF-IDF (fit) scores {triple(chk[REFERENCE])} and the k={K} neighbour vote "
          f"{triple(chk['knn_vote_k8'])}. The ticket set moves TF-IDF by at most {set_shift:.3f} against the de "
          f"slice here (that TF-IDF reads raw subject + body, so it differs slightly from the pilot's own classifier), "
          f"and German scores below English for most rows here, {verdict}"]
    return L


def render_md(res: dict) -> str:
    meta, pop, rows = res["meta"], res["population"], res["rows"]
    llm = {k: r for k, r in rows.items() if r["kind"] == "llm"}
    L = [
        "# Classifier comparison on val300",
        "",
        f"Generated by `scripts/llm_compare.py` at {meta['generated_utc']}. Every row is scored on the same "
        f"{meta['n']} val tickets (`{meta['selection']}`; ID set sha256 `{meta['ticket_ids_sha256'][:12]}`). "
        "Labels come from `kb_train`; `test_labels.parquet` is not read. Local models train on the `fit` "
        "partition only, which is also the LLM few-shot pool.",
        "",
        "These numbers are not comparable with the holdout TF-IDF baseline in `reports/`, which trains on the "
        "whole KB and scores different tickets.",
        "",
        "## Summary",
        "",
        "| row | type F1 | queue F1 | priority F1 | Incident+high recall | cost USD | p50 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        m = r["metrics"]
        rec = m["joint_incident_high"]["type_and_priority"]["recall"]
        L.append(f"| {r['label']} | " + " | ".join(f3(m["heads"][h]["macro_f1"]) for h in HEADS)
                 + f" | {fmt_rate(rec)} | {r['cost_usd']:.4f} | {ms(r['latency_ms_p50'])} |")
    L += ["",
          "Macro-F1 averages over labels present in gold or prediction (sklearn default, as in `llm_classify.py`). "
          "Incident+high recall uses the stricter rule (pred Incident and pred high). Local p50 is the batch "
          "mean per ticket, not a per-call latency.",
          "",
          "## Macro-F1 difference vs TF-IDF (paired bootstrap)",
          "",
          f"Same tickets resampled for both rows, {meta['bootstrap']['n_boot']} resamples, seed "
          f"{meta['bootstrap']['seed']}, 95% percentile CI. `*` marks a CI that excludes 0.",
          "",
          "| row | subset | type | queue | priority |",
          "|---|---|---|---|---|"]
    for name, r in rows.items():
        if name == meta["reference_row"]:
            continue
        for key, subset in (("bootstrap_vs_tfidf", f"all ({meta['n']})"),
                            ("bootstrap_vs_tfidf_non_twin", f"non-twins ({meta['n'] - pop['twins']})")):
            L.append(f"| {r['label']} | {subset} | " + " | ".join(ci(r[key][h]) for h in HEADS) + " |")

    L += ["", "## Accuracy per head", "", "| row | type | queue | priority |", "|---|---|---|---|"]
    for r in rows.values():
        L.append(f"| {r['label']} | " + " | ".join(fmt_rate(r["metrics"]["heads"][h]["accuracy"]) for h in HEADS) + " |")

    L += ["", "## Incident+high (major incident) detection", "",
          f"Gold events: {pop['gold_incident_high']} tickets with type Incident and priority high.", "",
          "| row | rule | flagged | recall | precision |", "|---|---|---|---|---|"]
    for r in rows.values():
        for rule, desc in JOINT_RULES.items():
            j = r["metrics"]["joint_incident_high"][rule]
            L.append(f"| {r['label']} | {desc} | {j['flagged']} | {fmt_rate(j['recall'])} | {fmt_rate(j['precision'])} |")

    L += ["", "## Calibration (10-bin ECE)", "",
          "Confidence is max predict_proba for TF-IDF, vote share for kNN (a coarse grid of eighths) and the "
          "model's normalised probability for its chosen label for LLM rows. The majority row has none.", "",
          "| row | type | queue | priority |", "|---|---|---|---|"]
    for r in rows.values():
        if r["metrics"]["ece"]:
            L.append(f"| {r['label']} | " + " | ".join(f3(r["metrics"]["ece"].get(h)) for h in HEADS) + " |")

    langs = pop["languages"]
    L += ["", "## Per detected language", "",
          "langdetect on the body, `DetectorFactory.seed = 0`. Population: "
          + ", ".join(f"{k} {v}" for k, v in sorted(langs.items())) + ". Small slices, so treat gaps under "
          "about 0.05 as noise.", "",
          "| row | lang | n | type | queue | priority |", "|---|---|---|---|---|---|"]
    for r in rows.values():
        for lang, s in r["metrics"]["language"].items():
            L.append(f"| {r['label']} | {lang} | {s['n']} | " + " | ".join(f3(s["macro_f1"][h]) for h in HEADS) + " |")

    L += german_md(res)

    copy = pop["twin_label_copy_accuracy"]
    L += ["", "## Paraphrase twins", "",
          f"A val ticket is a twin when its best word 1-2 gram TF-IDF cosine to any fit ticket is >= "
          f"{meta['twin_threshold']} (`twin_audit.py` method, vectorizer fit on the fit partition, subject + body). "
          f"Twins: {pop['twins']} of {meta['n']}. The twin is among the LLM's {meta['k']} few-shot examples for "
          f"{fmt_rate(pop['twin_in_fewshot'])} of them. Copying the twin's labels alone gets "
          + ", ".join(f"{h} {fmt_rate(copy[h])}" for h in HEADS) + " right. Twin slices are small, so their F1 "
          "is noisy.", "",
          "| row | twins F1 type / queue / priority | non-twins F1 type / queue / priority |", "|---|---|---|"]
    for r in rows.values():
        t, nt = r["metrics"]["twins"]["twin"], r["metrics"]["twins"]["non_twin"]
        L.append(f"| {r['label']} | " + " / ".join(f3(t["macro_f1"][h]) for h in HEADS)
                 + " | " + " / ".join(f3(nt["macro_f1"][h]) for h in HEADS) + " |")

    L += ["", "## Cost, tokens and latency", "",
          "| row | cost USD | USD per 1k tickets | prompt tok/ticket | output tok/ticket | thought tok/ticket "
          "| p50 ms | p95 ms | calls (cached) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows.values():
        if r["kind"] == "llm":
            t = r["tokens_per_call"]
            L.append(f"| {r['label']} | {r['cost_usd']:.4f} | {f3(r['cost_per_1k_tickets_usd'])} | "
                     f"{t['prompt']:.0f} | {t['output']:.0f} | {t['thought']:.0f} | {ms(r['latency_ms_p50'])} | "
                     f"{ms(r['latency_ms_p95'])} | {r['calls']} ({r['cached']}) |")
        else:
            L.append(f"| {r['label']} | 0 | 0 | n/a | n/a | n/a | {ms(r['latency_ms_p50'])} | "
                     f"{ms(r['latency_ms_p95'])} | local |")
    tf = rows[meta["reference_row"]]
    L += ["",
          f"TF-IDF fit took {tf['fit_seconds']} s on {meta['pool']}. LLM latency is per call under 16 concurrent "
          "workers and includes HTTP retries, so p95 reflects rate limiting as much as model speed. Tokens and "
          "cost count uncached calls only; a run resumed from cache under-reports both (see calls (cached)).",
          "", "## Comparability notes", "",
          "Held equal across rows:", "",
          f"- ticket IDs: the same {meta['n']}, asserted for every row (the script stops on any difference)",
          f"- training data / few-shot pool: {meta['pool']}; kb_extra is not used",
          f"- few-shot neighbours: k={meta['k']}, word 1-2 gram TF-IDF over greeting-stripped text; each LLM "
          "run's stored `fewshot_ids` are asserted equal to the rebuilt neighbours, and the kNN row votes over "
          "exactly that set (ties go to the most similar neighbour's label)"]
    if llm:
        pv = sorted({str(r["prompt_version"]) for r in llm.values()})
        L += [f"- prompt_version: {', '.join(pv)}; temperature 0 (set in `scripts/gemini_client.py`); "
              "structured JSON output with the same label enums"]
        L += ["", "Differs:", ""]
        for r in llm.values():
            price = r["price_per_1m_in_out_usd"]
            price_s = f"${price[0]:.2f} in / ${price[1]:.2f} out per 1M tokens" if price else "price unknown"
            L.append(f"- {r['model']}: thinking {r['thinking_level'] or 'default'}, {price_s}, run {r['generated_utc']}")
        L += ["- thinking level: minimal for gemini-3.5-flash-lite because that is its lowest; low for "
              "gemini-3.8-flash and gemini-4-carbon because they reject minimal (`reports/llm/smoke.md`)",
              "- model size and price: the models are not the same size, so a gap mixes capability and cost"]
    L += ["- local rows are deterministic; LLM rows are temperature 0 but not guaranteed bit-identical on rerun",
          ""]
    md = "\n".join(L)
    if any(ch in md for ch in (chr(0x2014), chr(0x2013))):
        raise CompareError("report contains an em or en dash")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", help="LLM run names under reports/llm; default: every val300_*.json")
    a = ap.parse_args()
    try:
        res = compare(a.runs)
    except CompareError as e:
        print(f"llm_compare failed: {e}", file=sys.stderr)
        return 2
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{OUT_NAME}.md").write_text(render_md(res), encoding="utf-8", newline="\n")
    (REPORTS / f"{OUT_NAME}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8", newline="\n")
    print(render_md(res).split("## Macro-F1 difference")[0])
    print(f"wrote reports/llm/{OUT_NAME}.md and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
