"""Near-duplicate audit between the KB split and the holdout.

MinHash LSH over word 5-gram shingles of the normalized body proposes candidate
pairs; every candidate is verified with exact Jaccard on the full shingle sets.
A chunked sparse all-pairs scan over the same shingles then measures LSH recall
for KB x holdout pairs.

Reads data/processed only. Uses kb_train labels; never reads test_labels.
Writes reports/near_dup_holdout_ids.txt and reports/near_dup_audit.md.

Run: .venv/Scripts/python scripts/dup_audit.py
"""
import hashlib
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
IDS_OUT = REPORTS / "near_dup_holdout_ids.txt"
MD_OUT = REPORTS / "near_dup_audit.md"

SHINGLE_WORDS = 5
NUM_PERM = 128
BANDS = 16
ROWS = 8
THRESHOLD = 0.85
SEED = 42
# a, b and x all range over the full field mod 2^31 - 1, so a*x + b < 2^62 never wraps uint64.
# A larger prime with a < 2^31 makes many permutations monotone in x (a*x + b < p) and
# collapses their minima onto the same shingle; that measured 83% LSH recall here.
PRIME = (1 << 31) - 1
LABELS = ["type", "queue", "priority", "language"]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def shingles(text: str, k: int = SHINGLE_WORDS) -> frozenset[str]:
    words = normalize(text).split()
    if len(words) < k:
        return frozenset([" ".join(words)])
    return frozenset(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def _shingle_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") % PRIME


def minhash_signatures(sets: list[frozenset], num_perm: int = NUM_PERM, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, PRIME, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, PRIME, size=num_perm, dtype=np.uint64)
    sigs = np.empty((len(sets), num_perm), dtype=np.uint64)
    for i, s in enumerate(sets):
        x = np.fromiter((_shingle_hash(t) for t in s), dtype=np.uint64, count=len(s))[:, None]
        sigs[i] = ((x * a + b) % np.uint64(PRIME)).min(axis=0)
    return sigs


def lsh_candidates(sigs: np.ndarray, bands: int = BANDS, rows: int = ROWS) -> set[tuple[int, int]]:
    assert bands * rows == sigs.shape[1]
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        chunk = np.ascontiguousarray(sigs[:, band * rows : (band + 1) * rows])
        for i in range(len(chunk)):
            buckets[chunk[i].tobytes()].append(i)
        for members in buckets.values():
            if len(members) > 1:
                pairs.update(combinations(members, 2))
    return pairs


def exact_cross_pairs(sets: list[frozenset], left: np.ndarray, right: np.ndarray, threshold: float = THRESHOLD, chunk: int = 500) -> set[tuple[int, int]]:
    """All (left, right) index pairs with Jaccard >= threshold, by sparse set intersection."""
    vocab: dict[str, int] = {}
    indptr, indices = [0], []
    for s in sets:
        indices.extend(vocab.setdefault(t, len(vocab)) for t in s)
        indptr.append(len(indices))
    x = sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.int32), np.array(indices), np.array(indptr)),
        shape=(len(sets), len(vocab)),
    )
    sizes = np.diff(x.indptr)
    xr_t = x[right].T.tocsc()
    found = set()
    for start in range(0, len(left), chunk):
        rows = left[start : start + chunk]
        inter = (x[rows] @ xr_t).toarray()
        union = sizes[rows][:, None] + sizes[right][None, :] - inter
        li, ri = np.nonzero(inter >= threshold * union)
        found.update(zip(rows[li].tolist(), right[ri].tolist()))
    return found


def main() -> None:
    t0 = time.perf_counter()
    kb = pd.read_parquet(PROC / "kb_train.parquet", columns=["ticket_id", "body", *LABELS])
    te = pd.read_parquet(PROC / "test_inputs.parquet", columns=["ticket_id", "body"])
    ids = np.array([*kb.ticket_id, *te.ticket_id])
    bodies = [*kb.body, *te.body]
    n_kb = len(kb)
    is_kb = np.arange(len(ids)) < n_kb

    sets = [shingles(b) for b in bodies]
    sigs = minhash_signatures(sets)
    cands = lsh_candidates(sigs)
    t_lsh = time.perf_counter() - t0

    verified = []
    for i, j in cands:
        sim = jaccard(sets[i], sets[j])
        if sim >= THRESHOLD:
            verified.append((i, j, sim))

    def kind(i: int, j: int) -> str:
        return {2: "kb-kb", 1: "kb-holdout", 0: "holdout-holdout"}[int(is_kb[i]) + int(is_kb[j])]

    cand_kinds = pd.Series([kind(i, j) for i, j in cands]).value_counts()
    ver = pd.DataFrame(verified, columns=["i", "j", "jaccard"])
    ver["kind"] = [kind(i, j) for i, j in zip(ver.i, ver.j)]
    ver_kinds = ver.kind.value_counts()

    # Orient cross pairs as (kb index, holdout index).
    cross = ver[ver.kind == "kb-holdout"].copy()
    cross[["kb", "ho"]] = np.sort(cross[["i", "j"]].to_numpy(), axis=1)
    flagged = sorted(set(ids[cross.ho]))

    t1 = time.perf_counter()
    exact = exact_cross_pairs(sets, np.arange(n_kb, len(ids)), np.arange(n_kb))
    lsh_cross = {(int(h), int(k)) for k, h in zip(cross.kb, cross.ho)}
    recall = len(lsh_cross & exact) / len(exact) if exact else 1.0
    exact_flagged = {ids[h] for h, _ in exact}
    t_exact = time.perf_counter() - t1

    kb_labels = kb[LABELS].to_numpy()
    kk = ver[ver.kind == "kb-kb"]
    agree = {c: float((kb_labels[kk.i, n] == kb_labels[kk.j, n]).mean()) if len(kk) else float("nan") for n, c in enumerate(LABELS)}
    agree_all = float((kb_labels[kk.i] == kb_labels[kk.j]).all(axis=1).mean()) if len(kk) else float("nan")

    # For holdout tickets with 2+ KB matches, do the matched KB tickets agree with each other?
    multi = cross.groupby("ho").kb.apply(list)
    multi = multi[multi.str.len() > 1]
    multi_agree = {c: float(np.mean([len(set(kb_labels[m, n])) == 1 for m in multi])) if len(multi) else float("nan") for n, c in enumerate(LABELS)}

    REPORTS.mkdir(exist_ok=True)
    IDS_OUT.write_text("".join(f"{t}\n" for t in flagged), encoding="utf-8")

    def example_row(r) -> str:
        return f"| {ids[r.kb]} | {ids[r.ho]} | {r.jaccard:.3f} | {normalize(bodies[r.kb])[:90]} | {normalize(bodies[r.ho])[:90]} |"

    cross_sorted = cross.sort_values(["jaccard", "kb", "ho"])
    picked = pd.concat([cross_sorted.head(3), cross_sorted.tail(2)]).drop_duplicates(subset=["kb", "ho"])
    examples = [example_row(r) for r in picked.itertuples()]
    rate = len(flagged) / len(te)
    lines = [
        "# Near-duplicate audit: KB vs holdout",
        "",
        "Generated by `scripts/dup_audit.py`. Inputs: `data/processed/kb_train.parquet`, `data/processed/test_inputs.parquet`. `test_labels.parquet` is not read.",
        "",
        "## Method",
        "",
        "1. Normalize each body: lowercase, collapse all whitespace runs to one space.",
        f"2. Shingle into word {SHINGLE_WORDS}-grams (a body under {SHINGLE_WORDS} words is one shingle).",
        f"3. MinHash: {NUM_PERM} permutations, universal hashing `(a*x + b) mod (2^31 - 1)` over 64-bit BLAKE2b shingle hashes reduced mod the same prime, numpy seed {SEED}.",
        f"4. LSH banding: {BANDS} bands x {ROWS} rows. Candidate probability at Jaccard 0.85 is {1 - (1 - 0.85 ** ROWS) ** BANDS:.4f}, at 0.5 it is {1 - (1 - 0.5 ** ROWS) ** BANDS:.4f}.",
        f"5. Verify every candidate with exact Jaccard on the full shingle sets; keep pairs with Jaccard >= {THRESHOLD}.",
        "6. Recall check: a chunked sparse all-pairs scan computes exact Jaccard for every KB x holdout pair and is compared with the LSH result.",
        "",
        "Many bodies share boilerplate greetings and sign-offs. Those shingles alone do not reach 0.85 because verification uses the whole body.",
        "",
        "## Results",
        "",
        f"- Documents: {n_kb} KB + {len(te)} holdout = {len(ids)}",
        f"- LSH candidate pairs: {len(cands)} ({', '.join(f'{k} {v}' for k, v in cand_kinds.items())})",
        f"- Verified pairs (Jaccard >= {THRESHOLD}): {len(ver)} ({', '.join(f'{k} {v}' for k, v in ver_kinds.items()) or 'none'})",
        f"- Verified KB x holdout pairs: {len(cross)}",
        f"- **Flagged holdout tickets** (at least one verified KB match): **{len(flagged)}** of {len(te)} ({rate:.2%})",
        f"- Exact all-pairs scan, KB x holdout: {len(exact)} pairs, {len(exact_flagged)} holdout tickets. LSH recall on those pairs: {recall:.2%}",
        "",
        "Flagged IDs: `reports/near_dup_holdout_ids.txt`, one per line.",
        "",
        "## Example KB x holdout pairs",
        "",
        "Lowest three and highest two Jaccard among verified pairs. Snippets are the first 90 normalized characters.",
        "",
        "| KB ticket | Holdout ticket | Jaccard | KB body | Holdout body |",
        "|---|---|---|---|---|",
        *(examples or ["| none | | | | |"]),
        "",
        "## Label agreement",
        "",
        "Holdout labels are not used. Agreement is measured on verified KB x KB pairs as a proxy for how much label signal a near-duplicate carries.",
        "",
        f"- Verified KB x KB pairs: {len(kk)}",
        *(f"- `{c}` agreement: {agree[c]:.2%}" for c in LABELS),
        f"- All four labels agree: {agree_all:.2%}",
        "",
        f"Holdout tickets with 2 or more verified KB matches: {len(multi)}.",
        *(
            ["", "Share of those where all matched KB tickets carry the same label:", "", *(f"- `{c}`: {multi_agree[c]:.2%}" for c in LABELS)]
            if len(multi)
            else []
        ),
        "",
    ]
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"candidates={len(cands)} verified={len(ver)} cross={len(cross)} flagged={len(flagged)} exact_cross={len(exact)} recall={recall:.4f}")
    print(f"runtime: lsh {t_lsh:.1f}s, exact scan {t_exact:.1f}s")
    print(f"wrote {IDS_OUT.relative_to(ROOT)} and {MD_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
