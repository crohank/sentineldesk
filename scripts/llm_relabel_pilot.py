"""Gate 1 of the German relabel plan: can Gemini reproduce this dataset's labels?

Labels 300 German KB tickets from the `val` partition, whose true labels are known, with a
Vertex AI Gemini model, and compares against a TF-IDF + LogisticRegression model fit on the
`fit` partition. If the LLM does not clearly beat that classifier on German, its labels for
the German file would add noise, and the relabel stops here (docs/D1_decisions.md).

Reads kb_train and kb_partitions only. Never reads test files.

    python scripts/llm_relabel_pilot.py --project sentineldesk-gdc-1121 --model gemini-3.8-flash

Auth: GOOGLE_ACCESS_TOKEN, else `gcloud auth print-access-token` (GCLOUD overrides the gcloud path).
Responses are cached in data/llm_cache/ keyed on model + prompt.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.text_normalize import ticket_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CACHE = ROOT / "data" / "llm_cache"
REPORT = ROOT / "reports" / "llm_relabel_pilot.md"
TARGETS = ["type", "queue", "priority"]
PROMPT_VERSION = "relabel-v1"
SEED = 42
DEFAULT_GCLOUD = Path.home() / "AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"


def token() -> str:
    if os.environ.get("GOOGLE_ACCESS_TOKEN"):
        return os.environ["GOOGLE_ACCESS_TOKEN"]
    gcloud = os.environ.get("GCLOUD", str(DEFAULT_GCLOUD))
    return subprocess.run([gcloud, "auth", "print-access-token"], capture_output=True, text=True,
                          check=True, shell=gcloud.endswith(".cmd")).stdout.strip()


def build_prompt(labels: dict, shots: pd.DataFrame, ticket: str) -> str:
    lines = [
        "You label customer support tickets for routing. Use exactly one value per field from these lists.",
        f"type: {', '.join(labels['type'])}",
        f"queue: {', '.join(labels['queue'])}",
        f"priority: {', '.join(labels['priority'])}",
        "in_scope: false only if the ticket is not a request to an IT or software company's support desk.",
        "Tickets may be in English or German. Labelled examples follow.",
        "",
    ]
    for _, s in shots.iterrows():
        lines += ["<example>", s["text"], f"-> type={s['type']} queue={s['queue']} priority={s['priority']}", "</example>"]
    lines += ["", "Label this ticket:", "<ticket>", ticket, "</ticket>"]
    return "\n".join(lines)


def make_call(project: str, location: str, model: str, labels: dict):
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    url = f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"
    schema = {"type": "OBJECT", "required": ["type", "queue", "priority", "in_scope"], "properties": {
        "type": {"type": "STRING", "enum": labels["type"]},
        "queue": {"type": "STRING", "enum": labels["queue"]},
        "priority": {"type": "STRING", "enum": labels["priority"]},
        "in_scope": {"type": "BOOLEAN"}}}
    auth = {"token": token(), "at": time.time()}

    def call(prompt: str) -> dict:
        key = hashlib.sha256(f"{model}|{PROMPT_VERSION}|{prompt}".encode()).hexdigest()
        path = CACHE / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                                     "responseSchema": schema}}
        for attempt in range(6):
            if time.time() - auth["at"] > 45 * 60:
                auth.update(token=token(), at=time.time())
            req = urllib.request.Request(url, json.dumps(body).encode(), {
                "Authorization": f"Bearer {auth['token']}", "Content-Type": "application/json"})
            try:
                r = json.load(urllib.request.urlopen(req, timeout=120))
                out = json.loads(r["candidates"][0]["content"]["parts"][0]["text"])
                out["_usage"] = r.get("usageMetadata", {})
                path.write_text(json.dumps(out), encoding="utf-8")
                return out
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
        raise RuntimeError("retries exhausted")
    return call


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="global")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    kb = pd.read_parquet(PROC / "kb_train.parquet").merge(pd.read_parquet(PROC / "kb_partitions.parquet"), on="ticket_id")
    kb["text"] = [ticket_text(s, b) for s, b in zip(kb["subject"], kb["body"])]
    labels = {t: sorted(kb[t].unique()) for t in TARGETS}

    DetectorFactory.seed = 0
    val = kb[kb["partition"] == "val"].copy()
    val["det"] = val["body"].map(lambda t: detect(t) if t.strip() else "??")
    pilot = val[val["det"] == "de"].sample(n=args.n, random_state=SEED).sort_values("ticket_id")

    fit = kb[kb["partition"] == "fit"]
    short = fit[fit["text"].str.split().str.len() < 70]
    short = short[short["body"].map(lambda t: detect(t) == "de")] if len(short) else short
    shots = short.groupby("queue").sample(n=1, random_state=SEED).sort_values("queue")

    call = make_call(args.project, args.location, args.model, labels)
    prompts = [build_prompt(labels, shots, t) for t in pilot["text"]]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(call, prompts))
    secs = time.perf_counter() - t0
    for t in TARGETS:
        pilot[f"llm_{t}"] = [r[t] for r in results]
    pilot["llm_in_scope"] = [r["in_scope"] for r in results]
    tok_in = sum(r["_usage"].get("promptTokenCount", 0) for r in results)
    tok_out = sum(r["_usage"].get("candidatesTokenCount", 0) for r in results)

    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)
    Xf = hstack([wv.fit_transform(fit["text"]), cv.fit_transform(fit["text"])]).tocsr()
    Xp = hstack([wv.transform(pilot["text"]), cv.transform(pilot["text"])]).tocsr()
    for t in TARGETS:
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(Xf, fit[t])
        pilot[f"clf_{t}"] = clf.predict(Xp)

    rows = []
    for t in TARGETS:
        for who in ["llm", "clf"]:
            rows.append((t, who, f1_score(pilot[t], pilot[f"{who}_{t}"], average="macro"),
                         accuracy_score(pilot[t], pilot[f"{who}_{t}"])))
    n = len(pilot)
    L = [
        "# Gemini relabel pilot (gate 1)",
        "",
        "Generated by `scripts/llm_relabel_pilot.py`. Reads `kb_train` and `kb_partitions` only.",
        "",
        f"- model: `{args.model}` on Vertex AI, location `{args.location}`, temperature 0, JSON schema output, prompt `{PROMPT_VERSION}`",
        f"- tickets: {n} from the `val` partition whose body langdetect calls German (seed {SEED})",
        f"- few-shot: one German `fit` ticket per queue ({len(shots)} examples), same prompt for every ticket",
        f"- classifier: TF-IDF word 1-2 + char_wb 3-5, LogisticRegression(C=4, balanced), fit on `fit` ({len(fit)} rows)",
        f"- tokens: {tok_in} in, {tok_out} out; wall time {secs:.0f}s with {args.workers} workers (cached calls count as 0s)",
        f"- LLM marked {int((~pilot['llm_in_scope']).sum())} of {n} as out of scope (all are real support tickets)",
        "",
        "| head | Gemini macro-F1 | classifier macro-F1 | Gemini accuracy | classifier accuracy |",
        "|---|---|---|---|---|",
    ]
    for t in TARGETS:
        llm = next(r for r in rows if r[0] == t and r[1] == "llm")
        clf = next(r for r in rows if r[0] == t and r[1] == "clf")
        L.append(f"| {t} | {llm[2]:.3f} | {clf[2]:.3f} | {llm[3]:.3f} ({int(round(llm[3] * n))}/{n}) | {clf[3]:.3f} ({int(round(clf[3] * n))}/{n}) |")
    L += ["", "Gemini vs classifier agreement with each other: " +
          ", ".join(f"{t} {(pilot[f'llm_{t}'] == pilot[f'clf_{t}']).mean():.3f}" for t in TARGETS), ""]
    REPORT.write_text("\n".join(L), encoding="utf-8", newline="\n")
    pilot[["ticket_id", *TARGETS, *[f"llm_{t}" for t in TARGETS], "llm_in_scope", *[f"clf_{t}" for t in TARGETS]]].to_parquet(
        CACHE / "pilot_results.parquet", index=False)
    print("\n".join(L))


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    main()
