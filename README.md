# SentinelDesk

Multilingual (en/de) ticket triage, escalation gate, and precedent-grounded reply drafting over the [Customer Support Tickets dataset](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (CC-BY-NC-4.0).

**Live demo:** [sentineldesk-wwnd44xj6q-uc.a.run.app](https://sentineldesk-wwnd44xj6q-uc.a.run.app) (console and API on Cloud Run)

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
python scripts/download_data.py   # data/raw/, pinned HF revision, SHA-256 checked
python scripts/make_split.py      # data/processed/ (locked 80/20, seed 42); fails on any mismatch
python scripts/make_partitions.py # data/processed/kb_partitions.parquet (fit/calib/val inside the KB)
python scripts/audit_20k.py       # data/processed/kb_extra.parquet (11,489 extra KB tickets, EXT- IDs)
python -m pytest -q               # split contract, leakage helpers, dup audit
python scripts/eda.py             # docs/eda_stats.txt, docs/figures/ (KB only)
```

The D1 branch provides shareable KB data, the split manifest, and test inputs. Generate the sealed test labels locally for evaluation.

### Console and API (uv)

```bash
uv sync
uv run serve        # http://127.0.0.1:8000  (POST /api/triage, GET /healthz)
uv run pytest
```

The console starts in fixture mode by default. For live Gemini triage with managed retrieval, see [the managed retrieval setup](deploy/retrieval/README.md). Shared request and response models live in `sentineldesk/schemas.py`. The exported JSON Schema is in `docs/schemas/contract.schema.json`.
## Evaluation

Prediction code writes one row per holdout ticket in the format in [docs/prediction_format.md](docs/prediction_format.md), then:

```bash
python scripts/score.py data/predictions/<run>.parquet --run-name <run> [--exclude-ids reports/near_dup_holdout_ids.txt]
```

Reports land in `reports/<run>/`. Baselines: `scripts/baseline_majority.py`, `scripts/baseline_tfidf.py`. Search indexes should call `scripts.leakage.assert_kb_only(ids)` before shipping.

## Leakage rule

Inference code reads only `kb_train.parquet` and `test_inputs.parquet`. Only the eval script may read `test_labels.parquet`. Holdout answers are never saved.

## Findings

See [docs/EDA.md](docs/EDA.md). Short version: use the main CSV only (the HF loader mixes in duplicates and an incompatible German file), the `language` column is wrong for about a quarter of `de` rows, and queue and priority are the hard targets (baseline macro-F1 0.59 and 0.63).
