"""Download the raw CSVs from Hugging Face into data/raw/."""
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Tobi-Bueck/customer-support-tickets"
FILES = [
    "aa_dataset-tickets-multi-lang-5-2-50-version.csv",  # used for the split
    "dataset-tickets-multi-lang-4-20k.csv",  # older version, 8,395 bodies overlap the main file
    "dataset-tickets-german_normalized_50_5_2.csv",  # different queue/priority taxonomy, no answer/type
    "README.md",
]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        print(hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW))


if __name__ == "__main__":
    main()
