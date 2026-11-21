"""Load the EmbeddingGemma vectors and KB evidence into the Qdrant vector DB (D3-22).

    python scripts/build_qdrant.py --scope full|fit

Reuses artifacts/dense/embeddinggemma/<scope>/ (no re-embedding) and the evidence of
artifacts/retrieval/<scope>/, so the database holds exactly what was evaluated.
Writes artifacts/qdrant/<scope>/ (embedded Qdrant storage).
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.vectordb import QdrantStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["fit", "full"], required=True)
    args = ap.parse_args()
    dense = ROOT / "artifacts" / "dense" / "embeddinggemma" / args.scope
    ids = np.load(dense / "ids.npy", allow_pickle=True)
    vectors = np.load(dense / "vectors.npy")
    sparse_ids = np.load(ROOT / "artifacts" / "retrieval" / args.scope / "ids.npy", allow_pickle=True)
    assert (ids == sparse_ids).all(), "dense and sparse indexes cover different IDs"
    ev = pd.read_parquet(ROOT / "artifacts" / "retrieval" / args.scope / "evidence.parquet").set_index("ticket_id").loc[ids]
    payloads = [{"ticket_id": t, "subject": str(r["subject"]), "answer": str(r["answer"]),
                 "answer_language": str(r["answer_language"])} for t, r in ev.iterrows()]
    out = ROOT / "artifacts" / "qdrant" / args.scope
    if out.exists():
        shutil.rmtree(out)  # our own build output; rebuilt from the vectors above
    out.mkdir(parents=True)
    t0 = time.time()
    store = QdrantStore.build(out, ids, vectors, payloads, embedding_model="embeddinggemma (Ollama)")
    print(f"{store.count()} points in '{store.collection}' at {out.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
