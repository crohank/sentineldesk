# SentinelDesk

Multilingual (English and German) support-ticket triage, escalation gate, and precedent-grounded reply drafting. Built by a five-person team in one day. The full plan, timeline and acceptance tests are in [docs/PLAN.md](docs/PLAN.md). Read it before making design decisions.

## Roles

| Dev | Owns | Code |
|---|---|---|
| D1 | Data split, leakage tests, scorer, reports | `scripts/`, evaluation |
| D2 | Triage classifier and calibration | `triage` component |
| D3 | Language detection, retrieval, evidence store | `retrieve`, `detect_language` |
| D4 | PII masking, escalation gate, drafting, validation | `mask`, `gate`, `draft`, `validate` |
| D5 | Schemas, API, console, packaging, deployment | `sentineldesk/schemas.py`, `pipeline.py`, `api.py`, `templates/`, `static/` |

Decision logs: D1 at [docs/D1_decisions.md](docs/D1_decisions.md), D5 at [decisions.md](decisions.md). Add an entry when you make a choice someone may ask about later.

## Commands

```bash
uv sync                    # Python 3.12-3.14; creates .venv
uv run serve               # console + API at http://127.0.0.1:8000
uv run pytest              # contract, safety and API tests
uv run export_schemas      # regenerate docs/schemas/contract.schema.json after editing schemas.py
```

`requirements.txt` is generated with `uv export --no-dev --no-hashes --no-emit-project -o requirements.txt`. Regenerate it after changing dependencies, so the plain pip path in the README keeps working.

## Layout

- `sentineldesk/schemas.py` holds the shared Pydantic contract. Field names match PLAN.md "Interfaces to freeze by 10:40". After any change, run `export_schemas` and commit the JSON file. A test fails if they drift.
- `sentineldesk/pipeline.py` has `analyze()`. The API, the console and batch evaluation all call it. It owns stage order, rule 1 (insufficient input), and rule 9 (a draft that fails validation becomes `review`/`DRAFT_FAILED`).
- `sentineldesk/components.py` wires the stage callables. Fixture stubs are used until real components land. To plug in a component, keep the signature and replace the stub in `load_components()`.
- `sentineldesk/fixtures/*.json` contains synthetic example responses: `routine`, `on_call` and `insufficient`. Every fixture string starts with `[FIXTURE]`.
- `scripts/` holds D1's data download, split and EDA scripts.

## Rules that must not break

- **Leakage.** Inference reads only `kb_train.parquet` and `test_inputs.parquet`. Only the scorer reads `test_labels.parquet`. Indexed IDs must be KB IDs (`TKT-00000`..`TKT-22856`). Holdout answers are never written to disk.
- **Do not trust the `language` column.** About 26% of rows labelled `de` are English. Detect language from text.
- **No hardcoded demo answers.** Never special-case a demo phrase. Fixtures are labelled as fixtures in the UI and in `retrieval_mode`/`generator_mode`.
- **Only validated drafts reach the response.** A draft whose citations are not retrieved precedents, or whose evidence span is not in the cited answer, is dropped.
- **Empty body is HTTP 200, not 422.** It returns `needs_clarification`. A missing subject is valid. Unknown request fields are rejected.
- **Offline console.** No CDN scripts or web fonts. Assets live in `sentineldesk/static/`. Use relative URLs.
- **Logs never contain ticket text, answers or prompts.** Log only IDs, labels, reason codes, modes and timings.
- **Label values are plain strings.** Queue names are learned from the KB.
- scikit-learn stays pinned to 1.8.0 (see D1-07). Python must be `>=3.12,<3.15`. D1 uses system Python 3.14 on Windows, and scikit-learn 1.8.0 has no 3.15 wheel.
- `analyze(ticket, mode="batch")` is the call for holdout prediction. A failure in `draft` or `validate` becomes `DRAFT_FAILED`, not a 500.

## Cloud

The team has full GCP access. Default to Cloud Run (images in Artifact Registry), Vertex AI (Gemma, Gemini, embeddings), Cloud Storage for artifact bundles, and Secret Manager for credentials. Every cloud call needs a local fallback, because the demo must also work without venue Wi-Fi.

## Data

Run `python scripts/download_data.py`, then `python scripts/make_split.py`. Data goes to `data/`, which is gitignored. The dataset is CC-BY-NC-4.0. Expected counts: 28,587 raw rows, 28,572 after dedup, 22,857 KB rows and 5,715 holdout rows.
