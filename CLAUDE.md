# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SentinelDesk: multilingual (en/de) support-ticket triage (type / queue / priority), an escalation gate, and precedent-grounded reply drafting (RAG over past tickets) on the Hugging Face `Tobi-Bueck/customer-support-tickets` dataset (CC-BY-NC-4.0). Current state is data prep + EDA only; no package, tests, or linter are configured yet.

## Commands

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.txt
python scripts/download_data.py   # -> data/raw/
python scripts/make_split.py      # -> data/processed/ (locked 80/20, seed 42)
python scripts/eda.py             # -> docs/eda_stats.txt, docs/figures/*.png (also prints baseline scores)
```

D2 (triage) currently works on a separate 80/10/10 split built from the official `load_dataset(...)` snippet (`datadownloadsnippet.txt`): rows lacking `type`/`answer` are dropped, then subject+body duplicates are removed before shuffling (seed 42). It does not overwrite D1's split.

```bash
python scripts/load_split_811.py  # -> data/split811/{train,val,test_inputs,test_labels}.parquet
python scripts/train_triage.py    # ~7 min; -> artifacts/triage/ (model + manifest), reports/triage/val_*
python scripts/eval_triage.py     # only reader of split811 test_labels; -> reports/triage/test_*
```

Triage lives in `sentineldesk/triage.py` (`TriagePredictor.load("artifacts/triage").predict(subject, body)` → `{type|queue|priority: {label, confidence, probabilities}}`). Temperatures and gate thresholds are stored in `artifacts/triage/manifest.json`.

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
