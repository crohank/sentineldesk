# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SentinelDesk: multilingual (en/de) support-ticket triage (type / queue / priority), an escalation gate, and precedent-grounded reply drafting (RAG over past tickets) on the Hugging Face `Tobi-Bueck/customer-support-tickets` dataset (CC-BY-NC-4.0). Code: `sentineldesk/` package (triage models, data loaders, embedding wrapper) plus `scripts/`; no tests or linter configured yet.

## Commands

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python scripts/download_data.py   # -> data/raw/
python scripts/make_split.py      # -> data/processed/ (locked 80/20, seed 42)
python scripts/eda.py             # -> docs/eda_stats.txt, docs/figures/*.png (also prints baseline scores)
```

### D2 triage on D1's split (current)

D2 trains on D1's canonical split from `origin/d1/data-eval`, copied (not merged) into the gitignored `data/processed/`: `kb_train.parquet` + `kb_partitions.parquet` (fit 18,285 / calib 2,286 / val 2,286; SHA checked in `sentineldesk/data_d1.py`) and `kb_extra.parquet` (20k-file extension; `load_kb_extra()` drops twins of calib/val tickets and adds the rest to fit only). Fit → train, calib → hyperparameters + temperatures, val → all reported metrics and rule-6 thresholds. Test files are never read.

```bash
git show origin/d1/data-eval:data/processed/kb_train.parquet > data/processed/kb_train.parquet   # same for kb_partitions, kb_extra
python -u scripts/train_triage_variants.py --features tfidf [--head svm|lgbm] [--extra] [--lang en] [--no-diagnostics] \
    > reports/triage_d1/logs/train_<variant>.log 2>&1      # live log: Get-Content -Wait <log>
python -u scripts/svm_search.py > reports/triage_d1/logs/svm_search.log 2>&1   # TF-IDF+LinearSVC grid -> reports/triage_d1/svm_search.md
python scripts/compare_triage.py --split d1               # all finished variants -> reports/triage_d1/compare.md
```

Variants live in `artifacts/triage_d1/<variant>/` (gitignored). `tfidf` = variant A (`TriagePredictor`); TF-IDF+SVM/LightGBM and EmbeddingGemma variants use `DenseTriagePredictor` (`sentineldesk/triage_dense.py`), which returns the same `{type|queue|priority: {label, confidence, probabilities}}` schema; temperatures and gate thresholds are in each `manifest.json`. Findings so far: EmbeddingGemma (`sentineldesk/embed.py`, cached under `artifacts/embeddings/`) is far behind TF-IDF even when standardized; kb_extra did not help LR; English-only training did not help; TF-IDF + LinearSVC + kb_extra is the first variant to pass the decision rule against A (see `reports/triage_d1/`).

The older split811 work (`load_split_811.py`, `train_triage.py`, `eval_triage.py`, `train_triage_dense.py`) is kept for reference only.

`data/` is gitignored; everyone regenerates the identical split from the scripts. Scripts resolve paths from the repo root via `Path(__file__).resolve().parents[1]`, so they can be run from any cwd.

## Data pipeline and leakage rule (critical)

`make_split.py` produces three files in `data/processed/`:
- `kb_train.parquet` (22,857 rows): all columns incl. `answer`, used for training and RAG indexing.
- `test_inputs.parquet` (5,715): `ticket_id, subject, body` only — the only test data inference code may read.
- `test_labels.parquet`: `ticket_id, type, queue, priority, language` — **only the eval script may read this.**

Holdout answers are never written to disk. Do not change the split logic (seed 42, dedup, shuffle, `TKT-xxxxx` ids) — results across the project depend on it being locked.

Use only `aa_dataset-tickets-multi-lang-5-2-50-version.csv`. Do not use `load_dataset(...)` on the HF repo: it concatenates all three CSVs, and the 20k file duplicates 8,395 main-file bodies (test leakage) while the German file has an incompatible queue/priority taxonomy and no `type`/`answer`.

## Data facts that should shape modeling (see docs/EDA.md)

- The `language` column is wrong for ~26% of `de` rows (English body). Detect language from the text; reply in the detected language.
- Bodies start with heavy greeting boilerplate — strip it before embedding, or retrieval matches on the greeting.
- 13.5% of subjects are empty; literal `\n` escapes are already converted in the split.
- Answers contain mask tokens (`<tel_num>`, `<name>`, `<acc_num>`, `[Name]`); generated drafts must not copy them verbatim.
- Answers almost never mention money, percentages, refunds, guarantees, or SLA hours (<0.5%), so any of these in a draft is unsupported by precedent (basis for a fabrication check).
- Escalation rule `priority==high & type==Incident` covers ~17% of tickets. Priority depends far more on queue than on type.
- Baseline (TF-IDF word+char + LogisticRegression, balanced): macro-F1 type 0.884, queue 0.587, priority 0.631; high-priority Incident recall 0.654. Queue and priority are the hard targets.

## Environment

Windows Smart App Control blocks some fresh compiled wheels on the dev machine: scikit-learn is pinned to 1.8.0 (1.9.x `_loss` DLL is blocked); Python 3.14 works, uv's Python 3.11 did not. Verify torch / faiss / sentence-transformers load before depending on them.
