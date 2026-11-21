# SentinelDesk

Multilingual (en/de) ticket triage, escalation gate, and precedent-grounded reply drafting over the [Customer Support Tickets dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (CC-BY-NC-4.0).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
python scripts/download_data.py   # data/raw/
python scripts/make_split.py      # data/processed/ (locked 80/20, seed 42)
python scripts/eda.py             # docs/eda_stats.txt, docs/figures/
```

Data files are gitignored. Everyone regenerates the same split from the scripts.

## Leakage rule

Inference code reads only `kb_train.parquet` and `test_inputs.parquet`. Only the eval script may read `test_labels.parquet`. Holdout answers are never saved.

## Findings

See [docs/EDA.md](docs/EDA.md). Short version: use the main CSV only (the HF loader mixes in duplicates and an incompatible German file), the `language` column is wrong for about a quarter of `de` rows, and queue and priority are the hard targets (baseline macro-F1 0.59 and 0.63).
