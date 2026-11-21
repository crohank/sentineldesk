"""Build a dense index (EmbeddingGemma or E5) over exactly the same IDs as the sparse index of that scope.

    python scripts/build_dense.py --scope fit|full [--backend embeddinggemma|e5]

Run scripts/build_index.py for the same scope first. Writes artifacts/dense/<backend>/<scope>/.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sentineldesk.dense import BACKENDS, DenseIndex, encode  # noqa: E402
from sentineldesk.textprep import clean_for_search  # noqa: E402

KB = ROOT / "data" / "processed" / "kb_train.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["fit", "full"], required=True)
    ap.add_argument("--backend", choices=list(BACKENDS), default="embeddinggemma")
    args = ap.parse_args()

    ids = np.load(ROOT / "artifacts" / "retrieval" / args.scope / "ids.npy", allow_pickle=True)
    assert (ids < "TKT-22857").all()
    kb = pd.read_parquet(KB).set_index("ticket_id").loc[ids]
    texts = [clean_for_search(s, b) for s, b in zip(kb["subject"], kb["body"])]
    t0 = time.time()
    vecs = encode(texts, "passage", args.backend)
    manifest = {"mode": "dense", "scope": args.scope, "backend": args.backend, "model": BACKENDS[args.backend]["model"], "n_docs": int(len(ids)),
                "dim": int(vecs.shape[1]), "build_seconds": round(time.time() - t0, 1)}
    DenseIndex(ids, vecs, manifest).save(ROOT / "artifacts" / "dense" / args.backend / args.scope)
    print(manifest)


if __name__ == "__main__":
    main()
