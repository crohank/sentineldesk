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

Release retriever: EmbeddingGemma query vectors (local Ollama) searched in a **Qdrant** vector DB (embedded, on disk, exact search), with in-memory search and sparse TF-IDF as automatic fallbacks ([D3-12](docs/D3_decisions.md), [D3-22](docs/D3_decisions.md)).

```bash
ollama serve &                                 # keep running while serving
ollama pull embeddinggemma                     # 621 MB, once per machine
python scripts/make_kb_partitions.py           # provisional fit/calib/valid partition of the KB
python scripts/build_index.py --scope full     # sparse index + evidence store -> artifacts/retrieval/full
python scripts/build_dense.py --scope full     # EmbeddingGemma index -> artifacts/dense/embeddinggemma/full (~5 min)
python scripts/build_qdrant.py --scope full    # load those vectors + answers into the Qdrant vector DB -> artifacts/qdrant/full
python -m pytest tests/test_retrieval.py
```

```python
from sentineldesk.retrieval import Retriever
r = Retriever.load("artifacts/retrieval/full", dense="artifacts/dense/embeddinggemma/full",
                   vectordb="artifacts/qdrant/full")     # or vectordb_url="http://localhost:6333"
r.warm_up()                                        # once at startup, loads EmbeddingGemma into memory
precedents, mode = r.retrieve_with_mode({"subject": None, "body": "..."}, k=5)
# mode: "qdrant:embeddinggemma"; "dense:embeddinggemma" if Qdrant fails; "sparse" if Ollama is down; "none" for a blank body
r.full_answer("TKT-00123")                         # full historical answer, for validation and the expanded view
```

Evaluation runs on the validation partition against fit-only indexes: `build_index.py --scope fit` and `build_dense.py --scope fit`, then `eval_retrieval.py --mode sparse|dense|hybrid`, `compare_retrieval.py` (paraphrase twins removed) and `eval_rerank.py`. E5 and the reranker need `pip install -r requirements-dense.txt`; serving does not. Decisions and measured numbers: [docs/D3_decisions.md](docs/D3_decisions.md). Example output for the console: [fixtures/precedents_example.json](fixtures/precedents_example.json).

## Grounded drafting (D3)

`sentineldesk/generation.py` drafts a reply from the retrieved precedents, in the ticket's language, citing them inline as `[TKT-#####]`. Backends are tried in order: **Gemini 3.5 Flash on Vertex AI** (when `GOOGLE_CLOUD_PROJECT` is set), **Gemma 4** locally through Ollama, then an extractive fallback that quotes evidence sentences verbatim and needs no model.

```bash
ollama pull gemma4:e2b-it-qat                      # 4.3 GB, needs Ollama >= 0.30.5
export GOOGLE_CLOUD_PROJECT=<project>              # optional; enables Gemini on Vertex
gcloud auth login                                  # Vertex calls use `gcloud auth print-access-token`
python scripts/eval_generation.py --n 40 --backend gemini   # or gemma | extractive | auto
python -m pytest tests/test_generation.py
```

```python
from sentineldesk.generation import Generator
gen = Generator.default()
res = gen.draft_with_mode(ticket, "de", precedents, min_top_score=0.35)  # 0.35 = EmbeddingGemma relevance guard
res.draft      # reply_text, cited_ticket_ids, claims[{text, ticket_id, evidence_span}], abstain (D5 Draft shape)
res.mode       # "gemini" | "gemma" | "offline_templates" | "none"
```

Every model draft passes deterministic checks before it is returned: cited IDs were retrieved, each evidence span is a verbatim substring of that precedent's answer, numbers appear in the evidence or the ticket, no new commitments (refund, guarantee, Erstattung, Gutschrift, ...), no placeholders, reply in the ticket's language. A failing draft gets one repair attempt, then the next backend runs.

## Vertex AI RAG Engine (D3)

```bash
python scripts/rag_engine_setup.py --project <project> --scope full --location europe-west3
```

Writes one file per KB ticket (same cleaned subject + body as the local indexes), uploads them to `gs://<project>-sentineldesk-kb`, creates a corpus with `text-multilingual-embedding-002`, imports in batches of at most 9,000 files, and records the corpus in `artifacts/rag_engine/full.json`. `RagEngineRetriever` maps each hit's source file back to its ticket ID and falls back to the local retriever if RAG Engine fails. New projects can't use RAG Engine in `us-central1`, `us-east1`, `us-east4` or `europe-west1` without an allowlist, hence `europe-west3`.
