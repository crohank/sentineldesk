"""Evaluate grounded drafting on KB validation tickets (D3-17).

    python scripts/eval_generation.py [--n 50] [--backend auto|gemini|gemma|extractive] [--min-score 0.35]

Retrieval uses the fit-only EmbeddingGemma index, so a validation ticket never retrieves itself.
Every draft goes through the same checks as production (generation.check_draft).
Metrics -> reports/generation_eval_<backend>.json; drafts -> artifacts/generation_samples_<backend>.jsonl
(ticket text stays out of git). Never reads holdout files.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.generation import GeminiVertex, GemmaOllama, Generator, check_draft  # noqa: E402
from sentineldesk.language import detect_language  # noqa: E402
from sentineldesk.retrieval import Retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--backend", choices=["auto", "gemini", "gemma", "extractive"], default="auto")
    ap.add_argument("--min-score", type=float, default=0.35)
    args = ap.parse_args()

    if args.backend == "auto":
        gen = Generator.default()
    elif args.backend == "gemini":
        gen = Generator([GeminiVertex()])
    elif args.backend == "gemma":
        gen = Generator([GemmaOllama()])
    else:
        gen = Generator([])
    label = "+".join(b.name for b in gen.backends) or "extractive"

    kb = pd.read_parquet(ROOT / "data" / "processed" / "kb_train.parquet").set_index("ticket_id")
    parts = pd.read_csv(ROOT / "splits" / "kb_partitions_provisional.csv")
    valid = kb.loc[parts.loc[parts["partition"] == "valid", "ticket_id"]].sample(args.n, random_state=7)
    r = Retriever.load(ROOT / "artifacts" / "retrieval" / "fit", dense=ROOT / "artifacts" / "dense" / "embeddinggemma" / "fit")

    rows, err_types = [], Counter()
    for tid, t in valid.iterrows():
        ticket = {"subject": t["subject"] or None, "body": t["body"]}
        lang, _ = detect_language(f"{t['subject']}\n{t['body']}")
        precedents, rmode = r.retrieve_with_mode(ticket, 5)
        t0 = time.perf_counter()
        res = gen.draft_with_mode(ticket, lang, precedents, min_top_score=args.min_score)
        ms = (time.perf_counter() - t0) * 1000
        d = res.draft
        final_errors = check_draft(d, lang, precedents, f"{t['subject']}\n{t['body']}") if not d.abstain else []
        reply_lang = detect_language(d.reply_text)[0] if not d.abstain else None
        for a in res.attempts:
            for e in a["errors"]:
                err_types[e.split(":")[0] if e.startswith("sentence") else e.split(" (")[0][:60]] += 1
        rows.append({
            "ticket_id": tid, "language": lang, "retrieval_mode": rmode,
            "top_score": max((p.score for p in precedents), default=None),
            "mode": res.mode, "abstain": d.abstain,
            "guard": any(a["backend"] == "relevance_guard" for a in res.attempts),
            "first_try_ok": bool(res.attempts) and res.attempts[0]["backend"] not in ("extractive", "relevance_guard")
                            and res.attempts[0]["ok"] and not d.abstain,
            "model_abstain": d.abstain and res.mode in ("gemini", "gemma"),
            "repaired": any(a["attempt"] == 2 and a["ok"] for a in res.attempts),
            "fallback": res.mode == "offline_templates",
            "final_valid": not final_errors, "reply_lang_match": (reply_lang in (lang, "und")) if reply_lang else None,
            "n_claims": len(d.claims), "words": len(d.reply_text.split()), "ms": round(ms, 1),
            "reply": d.reply_text, "cited": d.cited_ticket_ids, "attempts": res.attempts,
        })
        print(f"{tid} [{lang}] {res.mode:17} abstain={d.abstain!s:5} {ms:6.0f}ms", flush=True)

    df = pd.DataFrame(rows)
    drafted = df[~df["abstain"]]
    model_calls = df[~df["guard"] & df["language"].isin(["en", "de"])]
    lat = df.loc[~df["guard"], "ms"]
    res = {
        "backends": label, "n": len(df), "min_top_score": args.min_score, "retrieval": "embeddinggemma fit-only",
        "drafted": int(len(drafted)), "drafted_rate": round(len(drafted) / len(df), 3),
        "abstained_by_relevance_guard": int(df["guard"].sum()),
        "abstained_by_model": int(df["model_abstain"].sum()),
        "final_mode_counts": df["mode"].value_counts().to_dict(),
        "first_try_pass_rate_of_model_calls": round(float(model_calls["first_try_ok"].mean()), 3) if len(model_calls) else None,
        "repaired": int(df["repaired"].sum()), "fell_back_to_extractive": int(df["fallback"].sum()),
        "all_drafts_pass_checks": bool(drafted["final_valid"].all()) if len(drafted) else None,
        "reply_language_match_rate": round(float(drafted["reply_lang_match"].mean()), 3) if len(drafted) else None,
        "mean_claims_per_draft": round(float(drafted["n_claims"].mean()), 2) if len(drafted) else None,
        "mean_words_per_draft": round(float(drafted["words"].mean()), 1) if len(drafted) else None,
        "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1)} if len(lat) else None,
        "check_failures_during_attempts": dict(err_types.most_common(10)),
        "by_language": {l: {"n": int((df.language == l).sum()), "drafted": int(((df.language == l) & ~df.abstain).sum())}
                        for l in ["en", "de"]},
    }
    tag = args.backend
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"generation_eval_{tag}.json").write_text(json.dumps(res, indent=2))
    with open(ROOT / "artifacts" / f"generation_samples_{tag}.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
