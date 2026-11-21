"""Populate the versioned fit collection. Reruns reconcile against remote ticket IDs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from sentineldesk.agent_retrieval import AgentRetrieval
from sentineldesk.safety.pii import mask_pii


def remote_ids(store):
    ids = set()
    token = ""
    while True:
        page = store.post(store.objects + ":query", {"pageSize": 1000, "pageToken": token,
                                                    "outputFields": {"dataFields": ["ticket_id"]}})
        ids.update(obj["data"]["ticket_id"] for obj in page.get("dataObjects", []))
        token = page.get("nextPageToken")
        if not token:
            return ids


def build(data, workers):
    store = AgentRetrieval()
    kb = pd.read_parquet(data / "kb_train.parquet", columns=["ticket_id", "subject", "body", "answer"])
    partitions = pd.read_parquet(data / "kb_partitions.parquet")
    canonical = {f"TKT-{i:05d}" for i in range(22857)}
    if set(kb.ticket_id) != canonical or set(partitions.ticket_id) != canonical:
        raise ValueError("Unexpected KB or partition manifest")
    fit = kb.merge(partitions, on="ticket_id", validate="one_to_one").query("partition == 'fit'")
    fit = fit[fit.answer.fillna("").str.strip().ne("")]
    if len(fit) != 18280:
        raise ValueError("Unexpected resolved fit partition")
    existing = remote_ids(store)
    expected = set(fit.ticket_id)
    if existing - expected:
        raise ValueError("Collection contains unexpected IDs")
    todo = list(fit[~fit.ticket_id.isin(existing)].itertuples(index=False))
    print(f"{len(existing)} present, {len(todo)} to index", flush=True)

    def prepare(row):
        subject = mask_pii(row.subject).text
        body = mask_pii(row.body).text
        vector = store.embed(f"title: {subject.strip() or 'none'} | text: {body}")
        return {"parent": store.parent, "dataObjectId": row.ticket_id.lower(),
                "dataObject": {"vectors": {"embedding": {"dense": {"values": vector}}},
                               "data": {"ticket_id": row.ticket_id, "subject": subject,
                                        "answer": mask_pii(row.answer).text}}}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(todo), 100):
            chunk = todo[start:start+100]
            requests = list(pool.map(prepare, chunk))
            store.post(store.objects + ":batchCreate", {"requests": requests})
            if start % 1000 == 0:
                print(f"Stored {len(existing)+start+len(chunk)}", flush=True)
    if remote_ids(store) != expected:
        raise RuntimeError("Remote IDs differ from fit manifest; rerun to reconcile")
    print(f"Verified all {len(expected)} IDs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    build(args.data, args.workers)
