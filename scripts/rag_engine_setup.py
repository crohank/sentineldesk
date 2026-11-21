"""Build the Vertex AI RAG Engine corpus for the KB (D3-16).

    python scripts/rag_engine_setup.py --project PROJECT_ID [--scope full|fit] [--location europe-west3]

Needs `gcloud auth login` done. Steps (each is safe to re-run):
  1. write one file per KB ticket: artifacts/rag_engine/docs/<scope>/partN/<TICKET_ID>.txt
     (IDs come from artifacts/retrieval/<scope>/ids.npy, so no holdout ID can be uploaded)
  2. enable the Vertex AI and Cloud Storage APIs
  3. create the bucket gs://<project>-sentineldesk-kb (if missing) and rsync the files
  4. create the corpus with text-multilingual-embedding-002 (if no corpus is recorded yet)
  5. import each part into its own corpus, concurrently (<= 9,000 files each; API limit 10,000) and wait
Writes artifacts/rag_engine/<scope>.json with project, location, corpus and counts.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.rag_engine import API, GcloudToken, rest  # noqa: E402
from sentineldesk.textprep import clean_for_search  # noqa: E402

KB = ROOT / "data" / "processed" / "kb_train.parquet"
OUT = ROOT / "artifacts" / "rag_engine"
EMBEDDING_MODEL = "text-multilingual-embedding-002"
PART_SIZE = 9000


def sh(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def write_docs(scope: str) -> tuple[Path, int, int]:
    ids = np.load(ROOT / "artifacts" / "retrieval" / scope / "ids.npy", allow_pickle=True)
    assert (ids < "TKT-22857").all(), "holdout ID in the document set"
    kb = pd.read_parquet(KB).set_index("ticket_id").loc[ids]
    base = OUT / "docs" / scope
    n_parts = (len(ids) + PART_SIZE - 1) // PART_SIZE
    for i, (tid, row) in enumerate(kb.iterrows()):
        d = base / f"part{i // PART_SIZE}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{tid}.txt"
        text = clean_for_search(row["subject"], row["body"])
        if not f.exists() or f.read_text() != text:
            f.write_text(text, encoding="utf-8")
    print(f"{len(ids)} docs in {n_parts} parts under {base.relative_to(ROOT)}")
    return base, len(ids), n_parts


def upload(P: str, L: str, bucket: str, base: Path, scope: str) -> None:
    sh("gcloud", "services", "enable", "aiplatform.googleapis.com", "storage.googleapis.com", "--project", P)
    if sh("gcloud", "storage", "buckets", "describe", f"gs://{bucket}", "--project", P, check=False).returncode != 0:
        sh("gcloud", "storage", "buckets", "create", f"gs://{bucket}", "--project", P, "--location", L,
           "--uniform-bucket-level-access")
    sh("gcloud", "storage", "rsync", "--recursive", str(base), f"gs://{bucket}/{scope}", "--project", P)



def wait_op(name: str, location: str, project: str, token: GcloudToken, every: float = 10.0) -> dict:
    url = f"https://{location}-aiplatform.googleapis.com/{API}/{name}"
    while True:
        op = rest("GET", url, project, token)
        if op.get("done"):
            if "error" in op:
                raise RuntimeError(f"operation failed: {op['error']}")
            return op
        time.sleep(every)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--scope", choices=["full", "fit"], default="full")
    ap.add_argument("--location", default="europe-west3",
                    help="us-central1, us-east1, us-east4 and europe-west1 need an allowlist for new projects")
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--skip-upload", action="store_true", help="files are already in the bucket")
    args = ap.parse_args()
    P, L = args.project, args.location
    bucket = args.bucket or f"{P}-sentineldesk-kb"
    manifest_path = OUT / f"{args.scope}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    token = GcloudToken()

    base, n_docs, n_parts = write_docs(args.scope)

    if not args.skip_upload:
        upload(P, L, bucket, base, args.scope)

    corpus = manifest.get("corpus") if manifest.get("location") == L and manifest.get("project") == P else None
    if not corpus:
        url = f"https://{L}-aiplatform.googleapis.com/{API}/projects/{P}/locations/{L}/ragCorpora"
        op = rest("POST", url, P, token, {
            "display_name": f"sentineldesk-kb-{args.scope}",
            "description": "SentinelDesk historical KB tickets (subject + body), one file per ticket. KB split only.",
            "rag_embedding_model_config": {"vertex_prediction_endpoint": {
                "endpoint": f"projects/{P}/locations/{L}/publishers/google/models/{EMBEDDING_MODEL}"}},
        })
        done = wait_op(op["name"], L, P, token, every=5)
        corpus = done["response"]["name"]
        print("corpus:", corpus)
        manifest.update({"project": P, "location": L, "corpus": corpus, "embedding_model": EMBEDDING_MODEL,
                         "bucket": bucket, "scope": args.scope, "n_docs": n_docs,
                         "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    # One corpus per part: an import into a corpus that is already importing returns HTTP 400, and
    # one corpus ingests ~420 files/min, so parts go into separate corpora and import concurrently
    # (the API allows 3 concurrent imports per region). Retrieval queries all corpora and merges.
    imported = manifest.setdefault("imported_parts", {})
    corpora = manifest.setdefault("corpora", {})
    if "0" not in corpora:
        corpora["0"] = corpus                       # the first corpus holds part0
    ops = manifest.setdefault("in_flight_ops", {})
    if manifest.get("in_flight"):                   # resume an import started by an older run
        prev = manifest.pop("in_flight")
        ops[str(prev["part"])] = prev["op"]
    for i in range(n_parts):
        k = str(i)
        if k in imported or k in ops:
            continue
        if k not in corpora:
            url = f"https://{L}-aiplatform.googleapis.com/{API}/projects/{P}/locations/{L}/ragCorpora"
            op = rest("POST", url, P, token, {
                "display_name": f"sentineldesk-kb-{args.scope}-part{i}",
                "description": f"SentinelDesk KB tickets, part {i} of {n_parts}. KB split only.",
                "rag_embedding_model_config": {"vertex_prediction_endpoint": {
                    "endpoint": f"projects/{P}/locations/{L}/publishers/google/models/{EMBEDDING_MODEL}"}},
            })
            corpora[k] = wait_op(op["name"], L, P, token, every=5)["response"]["name"]
            print(f"corpus part{i}: {corpora[k]}", flush=True)
        url = f"https://{L}-aiplatform.googleapis.com/{API}/{corpora[k]}/ragFiles:import"
        ops[k] = rest("POST", url, P, token, {"import_rag_files_config": {
            "gcs_source": {"uris": [f"gs://{bucket}/{args.scope}/part{i}/"]},
            "rag_file_chunking_config": {"chunk_size": 512, "chunk_overlap": 0},
            "max_embedding_requests_per_min": 1000,
        }})["name"]
        print(f"import part{i}: {ops[k]}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    for k, name in sorted(ops.items()):
        t0 = time.time()
        done = wait_op(name, L, P, token, every=20)
        resp = done.get("response", {})
        imported[k] = {"imported": int(resp.get("importedRagFilesCount", 0)),
                       "failed": int(resp.get("failedRagFilesCount", 0)),
                       "skipped": int(resp.get("skippedRagFilesCount", 0)),
                       "waited_seconds": round(time.time() - t0)}
        print(f"part{k}: {imported[k]}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    manifest["in_flight_ops"] = {k: v for k, v in ops.items() if k not in imported}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total = sum(v["imported"] + v["skipped"] for v in imported.values())
    print(f"corpus {corpus}: {total}/{n_docs} files imported or already present")


if __name__ == "__main__":
    main()
