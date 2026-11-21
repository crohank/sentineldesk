"""Download the raw CSVs from Hugging Face into data/raw/ at a pinned revision."""
import argparse
import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Tobi-Bueck/customer-support-tickets"
# Dataset commit from 2026-06-28. The split in make_split.py is locked to the main CSV at this revision.
REVISION = "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb"
MAIN_CSV = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
MAIN_CSV_SHA256 = "f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0"
FILES = [
    MAIN_CSV,  # used for the split
    "dataset-tickets-multi-lang-4-20k.csv",  # older version, 8,395 bodies overlap the main file
    "dataset-tickets-german_normalized_50_5_2.csv",  # different queue/priority taxonomy, no answer/type
    "README.md",
]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=RAW, help=f"download directory (default {RAW})")
    out = ap.parse_args().out

    out.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        print(hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=out))

    got = sha256(out / MAIN_CSV)
    if got != MAIN_CSV_SHA256:
        sys.exit(
            f"SHA-256 mismatch for {out / MAIN_CSV}\n"
            f"  expected {MAIN_CSV_SHA256}\n"
            f"  observed {got}\n"
            "The locked split depends on this exact file. Do not run make_split.py on it."
        )
    print(f"{MAIN_CSV} sha256 ok ({got})")


if __name__ == "__main__":
    main()
