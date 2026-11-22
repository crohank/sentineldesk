"""Step 0: benchmark EmbeddingGemma before building anything.

256 split811/train tickets, triage.normalize() text, classification prompt, batch 32.
Reports tickets/sec, projected full-run time, and single-ticket latency p50/p95 (100 calls),
on each available device. If the build-device projection exceeds 25 min, retries with
256-dim Matryoshka truncation + max length 256. Writes reports/triage_compare/bench_embed.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentineldesk.embed import Embedder  # noqa: E402
from sentineldesk.triage import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "triage_compare"
BUDGET_MIN = 25.0
TARGET_COUNTS = {"d1_split_28572": 28_572, "split811_train_val_36240": 36_240}


def bench(device: str, max_len: int, dim: int | None, texts: list[str], singles: list[str]) -> dict:
    emb = Embedder(device=device, max_len=max_len, dim=dim, batch_size=32)
    emb.encode(texts[:32])  # warmup (CUDA kernels, tokenizer)
    if device == "cuda":
        import torch
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    v = emb.encode(texts)
    if device == "cuda":
        import torch
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    tps = len(texts) / secs
    lat = []
    for s in singles:
        t = time.perf_counter()
        emb.encode([s])
        lat.append((time.perf_counter() - t) * 1000)
    res = {
        "device": device, "max_len": max_len, "dim": emb.dim, "revision": emb.revision, "prompt": emb.prompt,
        "n": len(texts), "seconds": secs, "tickets_per_sec": tps,
        "projected_minutes": {k: n / tps / 60 for k, n in TARGET_COUNTS.items()},
        "single_latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "n": len(lat)},
        "vector_shape": list(v.shape), "norm_check": float(np.linalg.norm(v, axis=1).mean()),
    }
    print(json.dumps({k: res[k] for k in ["device", "max_len", "dim", "tickets_per_sec", "projected_minutes",
                                           "single_latency_ms"]}, indent=1))
    del emb
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    REP.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "split811" / "train.parquet").sample(361, random_state=0)
    texts = [normalize(s, b) for s, b in zip(df["subject"], df["body"])]
    texts, singles = texts[:256], texts[256:361]  # 5 warmup + 100 timed singles
    singles = singles[5:]

    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    build_device = devices[0]
    out = {"build_device": build_device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "torch": torch.__version__, "runs": []}
    for d in devices:
        out["runs"].append(bench(d, 512, None, texts, singles))

    primary = next(r for r in out["runs"] if r["device"] == build_device)
    worst = max(primary["projected_minutes"].values())
    out["primary_within_budget"] = worst <= BUDGET_MIN
    if worst > BUDGET_MIN:
        print(f"projection {worst:.1f} min > {BUDGET_MIN} -> retry dim=256, max_len=256")
        fb = bench(build_device, 256, 256, texts, singles)
        out["runs"].append(fb)
        out["fallback_within_budget"] = max(fb["projected_minutes"].values()) <= BUDGET_MIN
        if not out["fallback_within_budget"]:
            print("STOP: fallback still exceeds the 25-minute budget")
    (REP / "bench_embed.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
