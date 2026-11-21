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

## Retrieval (D3)

Release retriever: EmbeddingGemma through local Ollama, with sparse TF-IDF as automatic fallback ([D3-12](docs/D3_decisions.md)).

```bash
ollama serve &                                 # keep running while serving
ollama pull embeddinggemma                     # 621 MB, once per machine
python scripts/make_kb_partitions.py           # provisional fit/calib/valid partition of the KB
python scripts/build_index.py --scope full     # sparse index + evidence store -> artifacts/retrieval/full
python scripts/build_dense.py --scope full     # EmbeddingGemma index -> artifacts/dense/embeddinggemma/full (~5 min)
python -m pytest tests/test_retrieval.py
```

```python
from sentineldesk.retrieval import Retriever
r = Retriever.load("artifacts/retrieval/full", dense="artifacts/dense/embeddinggemma/full")
r.warm_up()                                        # once at startup, loads EmbeddingGemma into memory
precedents, mode = r.retrieve_with_mode({"subject": None, "body": "..."}, k=5)
# mode: "dense:embeddinggemma", "sparse" if Ollama was unreachable, "none" for a blank body
r.full_answer("TKT-00123")                         # full historical answer, for validation and the expanded view
```

Evaluation runs on the validation partition against fit-only indexes: `build_index.py --scope fit` and `build_dense.py --scope fit`, then `eval_retrieval.py --mode sparse|dense|hybrid`, `compare_retrieval.py` (paraphrase twins removed) and `eval_rerank.py`. E5 and the reranker need `pip install -r requirements-dense.txt`; serving does not. Decisions and measured numbers: [docs/D3_decisions.md](docs/D3_decisions.md). Example output for the console: [fixtures/precedents_example.json](fixtures/precedents_example.json).
