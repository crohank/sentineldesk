"""Gemma vs template drafts on KB validation tickets that qualify for a draft.

Writes reports/d4/gemma_check.md: how often Gemma's draft passed validation and the claim
check (otherwise the template was used), latency, and example drafts.
Triage and retrieval are the dev-only stand-ins.

    python scripts/d4/gemma_check.py [N]
"""
from __future__ import annotations

import statistics
import sys
import time

from _common import REPORTS, load_kb
from dev_standins import RetrievalStandIn, load_triage

from sentineldesk.safety import contracts as c
from sentineldesk.safety import gemma
from sentineldesk.safety.drafting import draft_with_mode
from sentineldesk.safety.gate import gate


def main(limit: int) -> None:
    kb = load_kb()
    fit, val = kb[kb["partition"] == "fit"], kb[kb["partition"] == "val"].reset_index(drop=True)
    tri, ret = load_triage(), RetrievalStandIn(fit)
    rows, examples = [], []
    for i in range(len(val)):
        if len(rows) >= limit:
            break
        masked = c.MaskedTicket(subject=val.at[i, "subject"], body=val.at[i, "body"])
        lang = "de" if val.at[i, "language"] == "de" else "en"
        precs = ret.search(masked.subject, masked.body)
        t = tri.predict([masked.subject], [masked.body])[0]
        if gate(masked, lang, t, precs).status != c.DRAFT_READY:
            continue
        t0 = time.perf_counter()
        d, mode = draft_with_mode(masked, lang, precs, generator=c.GEMMA)
        ms = (time.perf_counter() - t0) * 1000
        rows.append((val.at[i, "ticket_id"], lang, mode, ms))
        if mode == c.GEMMA and len(examples) < 3:
            examples.append((val.at[i, "ticket_id"], lang, d.reply_text))
    n = len(rows)
    used = [r for r in rows if r[2] == c.GEMMA]
    lat = sorted(r[3] for r in rows)
    L = ["# Gemma vs templates (KB validation tickets that qualify for a draft)", "",
         f"Backend: {gemma.backend()}; drafting model {gemma.model_name(gemma.DRAFT_MODEL)}, claim check {gemma.model_name(gemma.CHECK_MODEL)}; "
         f"prompt {gemma.PROMPT_VERSION}. Triage and retrieval: dev-only stand-ins.", "",
         f"- Tickets: {n} ({sum(r[1] == 'de' for r in rows)} German)",
         f"- Gemma draft used (passed validation and the claim check): {len(used)}/{n}",
         f"- Fell back to the template draft: {n - len(used)}/{n}",
         f"- Latency per draft: median {statistics.median(lat):.0f} ms, max {lat[-1]:.0f} ms" if lat else "- no tickets",
         "", "| ticket | language | generator | ms |", "|---|---|---|---|"]
    L += [f"| {tid} | {lang} | {mode} | {ms:.0f} |" for tid, lang, mode, ms in rows]
    L += ["", "## Example Gemma drafts", ""]
    for tid, lang, text in examples:
        L += [f"**{tid}** ({lang})", "", "```", text, "```", ""]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "gemma_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
