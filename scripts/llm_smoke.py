"""Step 1 of docs/D1_gemini_classifier_plan.md: check each candidate model before spending on a run.

Per model: is it in models.list, does one real classification call with the enum schema
work, which thinking levels it accepts, and whether logprobs come back. One KB ticket
is sent; no holdout data. Writes reports/llm/smoke.md.

  python scripts/llm_smoke.py                       # default candidates
  python scripts/llm_smoke.py --models gemini-3.8-flash
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gemini_client import GeminiClient, GeminiError, list_models  # noqa: E402
from scripts.llm_classify import PROC, REPORTS, build_prompt, load_labels, schema, text_of  # noqa: E402

CANDIDATES = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-4-carbon"]
LEVELS = ["minimal", "low"]


def try_call(model: str, prompt: str, sch: dict, level: str | None, extra: dict | None = None) -> str:
    try:
        c = GeminiClient("gemini", model, thinking_level=level)
        out = c.generate_json(prompt, sch, extra_config=extra)
        u = out["usage"]
        return (f"ok, {out['latency_ms']:.0f} ms, in {u.get('promptTokenCount', '?')} / out "
                f"{u.get('candidatesTokenCount', '?')} / thought {u.get('thoughtsTokenCount', 0)}, "
                f"labels {out['json'].get('type')} | {out['json'].get('queue')} | {out['json'].get('priority')}"
                + (f", avgLogprobs {out['avg_logprobs']:.3f}" if out.get("avg_logprobs") is not None else ""))
    except GeminiError as e:
        return f"FAILED: {str(e)[:200]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=CANDIDATES)
    a = ap.parse_args()

    try:
        available = {m["name"].removeprefix("models/") for m in list_models()}
    except GeminiError as e:
        print(f"smoke test failed: {e}", file=sys.stderr)
        return 2

    kb = pd.read_parquet(PROC / "kb_train.parquet")
    labels = load_labels(kb)
    sch = schema(labels)
    ex = kb.iloc[:3].assign(text=text_of(kb.iloc[:3]).to_numpy())
    prompt = build_prompt(text_of(kb.iloc[[3]]).iat[0], ex, labels)

    lines = ["# Gemini smoke test", "", f"Models visible to this key: {len(available)}", ""]
    for m in a.models:
        lines += [f"## {m}", "", f"- in models.list: {'yes' if m in available else 'NO'}"]
        if m not in available:
            lines.append("")
            continue
        lines.append(f"- default thinking: {try_call(m, prompt, sch, None)}")
        for lvl in LEVELS:
            lines.append(f"- thinking {lvl}: {try_call(m, prompt, sch, lvl)}")
        lines.append(f"- responseLogprobs: {try_call(m, prompt, sch, None, {'responseLogprobs': True})}")
        lines.append("")
    gem = sorted(x for x in available if x.startswith("gemini-"))
    lines += ["## All gemini models on this key", "", ", ".join(gem), ""]

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "smoke.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
