"""Embed split811 train (the KB) and val once with EmbeddingGemma; write the cache.

Usage: python scripts/embed_kb.py [--dim 256 --max-len 256] [--device cpu]
Never reads test files. Writes reports/triage_compare/embed_build.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk.embed import Embedder  # noqa: E402
from sentineldesk.triage_dense import embed_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "split811"
REP = ROOT / "reports" / "triage_compare"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    REP.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    emb = Embedder(max_len=args.max_len, dim=args.dim, device=args.device)
    load_s = time.perf_counter() - t0
    out = {"embedder": emb.config(), "device": emb.device, "model_load_seconds": load_s, "splits": {}}
    for name in ["train", "val"]:
        df = pd.read_parquet(DATA / f"{name}.parquet")
        t = time.perf_counter()
        vec, meta = embed_frame(emb, df)
        secs = time.perf_counter() - t
        out["splits"][name] = {"n": len(df), "shape": list(vec.shape), "key": meta["key"],
                               "cache_hit": meta["cache_hit"], "seconds": secs,
                               "encode_seconds": meta["encode_seconds"]}
        print(f"{name}: {vec.shape} key={meta['key']} cache_hit={meta['cache_hit']} {secs:.1f}s")
    out["total_build_seconds"] = sum(s["encode_seconds"] for s in out["splits"].values())
    print(f"one-time embedding build: {out['total_build_seconds'] / 60:.1f} min on {emb.device}")
    if not any(s["cache_hit"] for s in out["splits"].values()):
        (REP / "embed_build.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
