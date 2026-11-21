"""Draft the fixed 100-ticket generation audit sample (D1-21) and write a predictions file.

Runs each holdout input through mask -> language -> triage -> retrieval -> gate -> draft ->
validate with rule 9, using the dev-only stand-ins for triage and retrieval (retrieval over
the full KB). Reads only test_inputs and kb data, never labels. The output follows
docs/prediction_format.md (draft columns included) so D1's judge_drafts.py can grade it.

    python scripts/d4/generation_audit.py SAMPLE_IDS.txt OUT.parquet [--generator gemma]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from _common import DATA, load_kb
from dev_standins import load_retrieval, load_triage
from langdetect import DetectorFactory, LangDetectException, detect

from sentineldesk.safety import contracts as c
from sentineldesk.safety import draft_with_mode, gate, mask_ticket, validate
from sentineldesk.safety.patterns import has_usable_text

DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "und"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sample_ids", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--generator", default=c.OFFLINE_TEMPLATES, choices=[c.OFFLINE_TEMPLATES, c.GEMMA])
    args = ap.parse_args()

    ids = [s.strip() for s in args.sample_ids.read_text().splitlines() if s.strip() and not s.startswith("#")]
    inputs = pd.read_parquet(DATA / "test_inputs.parquet").set_index("ticket_id").loc[ids].reset_index()
    tri, ret = load_triage(), load_retrieval()
    kb_ids = set(load_kb()["ticket_id"])
    rows = []
    for r in inputs.itertuples():
        t0 = time.perf_counter()
        masked = mask_ticket(c.TicketInput(body=r.body or "", subject=r.subject or None))
        language, triage, precs, d, mode = "und", None, [], None, None
        if not has_usable_text(masked.body):
            decision = gate(masked, "und", None, [])
        else:
            language = detect_language(masked.text)
            triage = tri.predict([masked.subject], [masked.body])[0]
            precs = ret.search(masked.subject, masked.body, k=5)
            decision = gate(masked, language, triage, precs)
            if decision.status == c.DRAFT_READY:
                d, mode = draft_with_mode(masked, language, precs, generator=args.generator)
                v = validate(d, language, precs, ticket=masked)
                if d.abstain or not v.valid:
                    decision = c.Decision(status=c.REVIEW, destination=c.HUMAN_REVIEW, reasons=[
                        c.Reason(code=c.DRAFT_FAILED, message="Draft withheld: " + "; ".join(v.errors))])
                    d = None
        assert all(p.ticket_id in kb_ids for p in precs)
        row = {
            "ticket_id": r.ticket_id,
            "decision_status": c.SCORER_STATUS[decision.status],
            "destination": decision.destination,
            "reason_codes": [x.code for x in decision.reasons],
            "retrieved_ids": [p.ticket_id for p in precs],
            "cited_ids": list(d.cited_ticket_ids) if d else [],
            "draft_text": d.reply_text if d else "",
            "draft_abstain": d is None,
            "draft_claims": [{"text": cl.text, "ticket_id": cl.ticket_id, "evidence_span": cl.evidence_span}
                             for cl in d.claims] if d else [],
            "generator": mode or "none",
            "language": language,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
        if triage is not None:
            for h in ("type", "queue", "priority"):
                row[f"pred_{h}"] = getattr(triage, h).label
                row[f"conf_{h}"] = getattr(triage, h).confidence
        rows.append(row)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"{len(df)} tickets, drafts {int((~df.draft_abstain).sum())}, statuses {df.decision_status.value_counts().to_dict()}")
    print(f"generators: {df.generator.value_counts().to_dict()}; median latency {df.latency_ms.median():.0f} ms")


if __name__ == "__main__":
    main()
