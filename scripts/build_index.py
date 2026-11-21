"""Build the retrieval index and evidence store.

    python scripts/build_index.py --scope fit    # fit partition only, for validation (D3-07)
    python scripts/build_index.py --scope full   # whole KB, release index

Reads only kb_train.parquet (and the partition file for --scope fit).
Writes artifacts/retrieval/<scope>/.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.language import detect_language  # noqa: E402
from sentineldesk.retrieval import Retriever  # noqa: E402
from sentineldesk.textprep import normalize  # noqa: E402

KB = ROOT / "data" / "processed" / "kb_train.parquet"
PARTS = ROOT / "splits" / "kb_partitions_provisional.csv"
OUT = ROOT / "artifacts" / "retrieval"
EVIDENCE_CACHE = ROOT / "artifacts" / "evidence_all.parquet"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_store(kb: pd.DataFrame) -> pd.DataFrame:
    """Every KB row's subject, cleaned answer, detected answer language, and labels for proxy metrics."""
    if EVIDENCE_CACHE.exists():
        ev = pd.read_parquet(EVIDENCE_CACHE)
        if len(ev) == len(kb) and set(ev["ticket_id"]) == set(kb["ticket_id"]):
            return ev.set_index("ticket_id")
    t0 = time.time()
    ev = pd.DataFrame({
        "ticket_id": kb["ticket_id"],
        "subject": kb["subject"].map(normalize),
        "answer": kb["answer"].map(normalize),
        "type": kb["type"], "queue": kb["queue"], "priority": kb["priority"],
    })
    ev["answer_language"] = ev["answer"].map(lambda a: detect_language(a)[0])
    ev["body_language"] = kb["body"].map(lambda b: detect_language(b)[0])
    print(f"language detection on {len(ev)} rows: {time.time() - t0:.0f}s")
    EVIDENCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(EVIDENCE_CACHE, index=False)
    return ev.set_index("ticket_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["fit", "full"], required=True)
    args = ap.parse_args()

    kb = pd.read_parquet(KB)
    ev = evidence_store(kb)
    docs = kb[kb["answer"].str.strip() != ""]  # D3-06
    if args.scope == "fit":
        parts = pd.read_csv(PARTS)
        fit_ids = set(parts.loc[parts["partition"] == "fit", "ticket_id"])
        docs = docs[docs["ticket_id"].isin(fit_ids)]

    t0 = time.time()
    r = Retriever.build(docs[["ticket_id", "subject", "body"]], ev, args.scope)
    r.manifest.update({
        "build_seconds": round(time.time() - t0, 1),
        "kb_sha256": sha256(KB),
        "partition_file": PARTS.name if args.scope == "fit" else None,
        "excluded_empty_answers": int((kb["answer"].str.strip() == "").sum()),
    })
    out = OUT / args.scope
    r.save(out)
    print(r.manifest)
    print("->", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
