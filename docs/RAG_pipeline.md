# SentinelDesk RAG pipeline on Google Cloud

How a ticket becomes a grounded reply, which products run each step, and where the retrieval move to Google Cloud stands. Status as of 2026-11-21, 1:50 PM ET. Retrieval numbers are D1's measurements; the D3 baseline numbers come from `docs/D3_decisions.md` on `d3/retrieval`.

## Request path

| # | Stage | Product / stack | Owner | Status |
|---|---|---|---|---|
| 1 | Ticket in | D5 API + Material 3 console | D5 | on `ajin-frank-justin` |
| 2 | PII masking | D4 safety module | D4 | on `D4` |
| 3 | Language detection | `langdetect`, seeded, abstains below 0.70 (D3-02) | D3 | on `d3/retrieval` |
| 4 | Triage labels | Calibrated TF-IDF + logistic regression | D2 | on `D2` |
| 5 | Retrieve top-5 precedents | **Gemini Embedding 2 (Vertex AI) + Vertex AI Agent Retrieval** | D3 | proposed, being loaded (below) |
| 6 | Escalation gate | D4 rules | D4 | on `D4` |
| 7 | Grounded draft | Gemini 3.5 Flash on Vertex AI, then Gemma 4 (local Ollama), then extractive quotes | D3 | on `d3/retrieval` |
| 8 | Grounding checks | Cited IDs retrieved, spans verbatim, numbers in evidence, no new commitments, language match (D3-17) | D3 | on `d3/retrieval` |
| 9 | Console | Draft + evidence cards | D5 | on `ajin-frank-justin` |

Fallbacks: if the cloud retrieval call fails, times out or returns nothing, retrieval falls back to the local index (EmbeddingGemma on Ollama with NumPy search, then sparse TF-IDF). If Gemini drafting fails, Gemma 4 runs locally, then extractive quotes. The demo still works offline.

Open conflict: the `D4` branch's latest commits make Gemma on Vertex AI the live drafting model, while D3-15/D3-18 put drafting under D3 with Gemini 3.5 Flash first. D3 and D4 need to pick one before integration.

## Retrieval: what changed and why

D3's branch ships EmbeddingGemma (local Ollama) with in-memory NumPy search. Vertex AI RAG Engine was coded but never loaded: on D3's project, the import calls Vertex embedding models at up to 1,000 per minute and the quota is 5 per minute (D3-16).

The proposal keeps RAG (retrieve, then generate grounded in the evidence) and moves both halves of retrieval to managed Google Cloud services:

1. **Embeddings: Gemini Embedding 2** (`gemini-embedding-2`, global endpoint, `:embedContent`, 768 dimensions, `taskType` RETRIEVAL_DOCUMENT for KB tickets and RETRIEVAL_QUERY for live tickets). On project `sentineldesk-gdc-1121` it has no 5-per-minute wall.
2. **Vector store: Vertex AI Agent Retrieval** (formerly Vector Search 2.0), collection `sd-kb` in `us-central1`. We upload our own vectors and do not use the collection's built-in embedding, which would call regional embedding models and could hit the old quota again. Each object carries `ticket_id`, `source`, `subject`, the exact `answer` (D3-14), detected `language` and `answer_language`, `queue`, `type`, `priority`.

Text sent for embedding is D3's `clean_for_search(subject, body)` (greetings and sign-offs stripped, D3-04). KB = `kb_train` (22,857) + `kb_extra` (11,489) minus 7 empty answers = 34,339 objects. Holdout tickets are never embedded or uploaded.

## Measurements

### Embedding throughput (Gemini Embedding 2)

- Pilot: 256 tickets in 2.6 s with 32 parallel requests, no 429s.
- Full run: 36,632 calls (34,346 KB docs + 2,286 validation queries) in 382 s at about 96 per second, 0 errors.
- Query latency: p50 0.25 s, p95 0.39 s per call.

For comparison, EmbeddingGemma on an 8-vCPU Compute Engine VM embeds 8 tickets per second (about 70 minutes for this KB). The project has no GPU quota (`GPUS_ALL_REGIONS` = 0).

### Retrieval quality, same protocol as D3

2,286 provisional `valid` queries against the 18,280-ticket fit-only index, exact search, label agreement of retrieved neighbours (a proxy for relevance, D3-10).

| | queue top-1 | queue any-of-5 | type top-1 | priority top-1 |
|---|---|---|---|---|
| EmbeddingGemma (D3) | **0.777** | **0.906** | **0.926** | **0.801** |
| Gemini Embedding 2 | 0.764 | 0.895 | 0.909 | 0.791 |

EmbeddingGemma leads by 1 to 2 points. On queue that is about 1.4 standard errors, so close to noise. By language, Gemini scores en 0.793 / de 0.697 queue top-1 (EmbeddingGemma 0.805 / 0.721).

The no-twin comparison (D3-09) is not like for like: D3's twin set also used E5 and EmbeddingGemma flags that are not available here, so this run removes 1.3 twins per query against D3's 3.3. Gemini's no-twin queue top-1 of 0.472 (vs 0.437) is on an easier set and should not be read as a win.

Full output: `reports/retrieval_gemini_embedding_2.json`.

### Relevance guard threshold

Gemini Embedding 2 scores sit closer together than EmbeddingGemma's, so D3's 0.35 guard (D3-19) does not carry over.

| text | best KB score |
|---|---|
| "What is the weather in Paris tomorrow?" | 0.596 |
| "Write me a poem about cats" | 0.637 |
| Short real tickets (printer, VPN, refund) | 0.78 to 0.79 |
| Validation queries, 5th percentile | 0.865 |

Starting point for the guard and D4's weak-evidence rule in this mode: about 0.70. Five probes only; tune on authored tickets before relying on it.

### Vector store speed

| | search latency | top-3 matches exact search |
|---|---|---|
| Agent Retrieval, 500-vector test collection | 0.3 to 4.3 s | 5/5 |
| Qdrant on Compute Engine, same 500 vectors | about 120 ms | 5/5 |

The full test (34k vectors, ScaNN index, 50 timed queries, recall@5 against exact search) runs as soon as the import below finishes.

## Status

- Gemini Embedding 2 vectors for the full KB: **done**.
- Quality comparison: **done** (table above).
- Agent Retrieval load: **in progress**. API writes are capped at 1,200 objects per minute per collection, so the load goes through bulk import from `gs://sentineldesk-gdc-1121-agent-retrieval`. Two imports failed on file format (bulk import wants `id` instead of `dataObjectId`, and `{"values": [...]}` instead of `{"dense": {"values": [...]}}`). Third import started 1:46 PM ET. 2,000 objects are already in from the API.
- Next: ScaNN index (`emb`, filter fields `language`, `answer_language`, `source`). Searches without an index use KNN, capped at 120 requests per minute. Then the latency and recall test.
- Decision rule: keep Agent Retrieval if search p95 is under about 1 s. Otherwise fall back to Qdrant on Compute Engine with EmbeddingGemma.
- D3 code change still needed: an `AgentRetrievalRetriever` with the same `retrieve_with_mode` interface, falling back to local retrieval on error, timeout (about 2 s) or an empty result.

## Google Cloud resources (project `sentineldesk-gdc-1121`)

| resource | purpose | keep? |
|---|---|---|
| Agent Retrieval collection `sd-kb` (us-central1) | release KB index | yes |
| Agent Retrieval collection `sd-e2e` | 500-vector speed test | delete |
| Bucket `gs://sentineldesk-gdc-1121-agent-retrieval` | bulk import files | delete after load |
| VM `sd-embed` (n2-standard-8, us-central1-a, about $0.39/hour) | EmbeddingGemma + Qdrant, side A of the comparison | until the latency test is done |
| Firewall `sd-embed-allow-laptop` | VM ports 11434/6333 open to one IP; Ollama and Qdrant have no auth | delete with the VM |

APIs enabled: `aiplatform`, `vectorsearch`, `compute`.

## Known issues on `d3/retrieval`

- `D3.load()` switches to RAG Engine whenever `artifacts/rag_engine/full.json` exists and `GOOGLE_CLOUD_PROJECT` is set. The setup script writes that manifest before the (failed) import, and an empty corpus returns zero hits without raising, so `sentineldesk/rag_engine.py:113` never falls back. Every ticket would get no precedents and drafting would abstain. Delete the manifest or pass `use_rag_engine=False`.
