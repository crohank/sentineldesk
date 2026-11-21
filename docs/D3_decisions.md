# D3 decision log (retrieval and language)

One entry per decision. Newest at the bottom. Each says what was decided and why, so a judge question or a teammate can be answered from here.

## 2026-11-21

### D3-01 D3 work on its own branch, same data as D1
**Decision:** All D3 work goes on branch `d3/retrieval` and lands through a PR (D1-13). D3 uses D1's split unchanged.
**Why:** Shared repo, reviewed `main`. Rebuilt the split locally with `download_data.py` + `make_split.py`: 22,857 KB (`TKT-00000`..`TKT-22856`), 5,715 holdout (`TKT-22857`..`TKT-28571`), source CSV SHA-256 `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`, same as D1-12. Hugging Face dataset revision at download time: `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb` (last modified 2026-06-28). Passed to D1 for pinning.

### D3-02 Detect language from text, never from the column
**Decision:** `sentineldesk.language.detect_language(text)` returns `en`, `de`, another ISO code, or `und`, plus a probability. It uses `langdetect` with a fixed seed. Text with fewer than 3 letters-only words, or whose top probability is below 0.70, is `und`.
**Why:** About 26% of `de`-labelled rows have English bodies (D1-08). The brief requires detection from raw text. `langdetect` is non-deterministic unless seeded, and on very short text it guesses, so we abstain rather than force English or German.

### D3-03 Search on subject + body only; look up the answer after
**Decision:** The index is built from historical subject + body. The answer is fetched from the evidence store by ticket ID after retrieval.
**Why:** A live ticket has no answer, so queries and documents must be the same kind of text. Keeps answer text out of similarity scoring.

### D3-04 Strip greeting and sign-off boilerplate before indexing and querying
**Decision:** `sentineldesk.textprep.clean_for_search` removes leading salutations ("Dear Customer Support Team,", "Sehr geehrter Kundenservice,") and trailing courtesy sentences ("Thank you.", "Vielen Dank.", "I look forward to hearing from you soon.", "[Your Name]"), `<br>` tags and mask tokens. The same function runs on KB documents and live queries.
**Why:** Most common body openings and closings are these exact sentences (e.g. "Dear Customer Support Team," starts 320 KB bodies, "Thank you." ends 353). D1 found 511 holdout bodies sharing their first 80 characters with a KB ticket while labels mostly differ. Left in, search matches on the greeting.

### D3-05 Answer language is detected per answer
**Decision:** The evidence store records `answer_language` from `detect_language(answer)`, not the `language` column.
**Why:** Same noise as D3-02. The drafting step and the console need to know the real language of the evidence to claim cross-lingual grounding honestly.

### D3-06 Tickets with empty answers are not indexed
**Decision:** The 5 KB rows with an empty answer are left out of the retrieval index.
**Why:** A precedent without a resolution cannot ground a draft and only takes a top-5 slot.

### D3-07 Validate against a fit-only index; release index covers the full KB
**Decision:** Retrieval is scored with validation-partition tickets as queries against an index of the fit partition only. The release index is rebuilt over the full KB (minus D3-06). Every build asserts that indexed IDs are KB IDs and none is a holdout ID.
**Why:** Scoring KB tickets against an index that contains them would return themselves. Holdout tickets are never used to choose retrieval settings.

### D3-08 Provisional internal KB partition
**Decision:** Until D1 publishes theirs, `scripts/make_kb_partitions.py` writes `splits/kb_partitions_provisional.csv`: seeded (42) 80/10/10 fit / calib / valid, stratified by type × priority. Holdout IDs are never touched.
**Why:** D3 needs a validation set now to choose the retrieval mode. Uses the recipe from the plan so D1's version should match in spirit; will be replaced when D1's manifest lands.

### D3-09 The KB is full of paraphrase twins (finding, affects D1, D2, D4)
**Decision:** Treat a sparse similarity of 0.5 or more as "paraphrase twin" and report retrieval both with and without twins.
**Why:** The generator produced the same ticket several times with small wording changes ("Several Tools Experiencing Failures" vs "...Malfunctions", score 0.88). On the validation partition, queue agreement of the top-1 neighbour is 0.29 below score 0.3, 0.60 at 0.4–0.5, 0.85 at 0.5–0.6 and 0.97–1.00 above 0.6. Using `test_inputs` only (no labels), 58.9% of holdout tickets have a KB neighbour at 0.5 or more, 46.3% at 0.6 or more, 16.4% at 0.8 or more (`reports/holdout_twin_scores_sparse.csv`). Consequences:
- D1: holdout scores will be higher than on genuinely new tickets. The MinHash Jaccard 0.85 audit will miss most twins because paraphrases change too many 5-grams. The CSV above gives a ready-made twin flag per holdout ID for the "scores excluding near-duplicates" report.
- D2: a nearest-neighbour vote would score very high on the holdout for this reason, not because it generalizes. If used, report it with and without twins.
- D4: validation similarity scores are inflated. Authored, judge-style tickets score 0.12–0.29 top-1, against a validation median of 0.53. A weak-evidence threshold tuned on validation would escalate almost every live ticket. Tune it on authored tickets or on the no-twin setting.

### D3-10 Release retrieval mode: sparse
**Decision:** Ship sparse TF-IDF (word 1–2 + char_wb 3–5) as the release retriever. Keep dense (`multilingual-e5-small`) and hybrid (reciprocal rank fusion) built and scriptable as options, not in the default serving path.
**Why:** Measured on 2,286 validation queries against the fit-only index (`reports/retrieval_*.json`, `reports/retrieval_no_twin.json`):

| mode | queue top-1 | queue any-of-5 | type top-1 | priority top-1 | no-twin queue top-1 | p95 latency |
|---|---|---|---|---|---|---|
| sparse | 0.708 | 0.871 | 0.891 | 0.755 | 0.342 | 47 ms |
| dense | 0.679 | 0.861 | 0.886 | 0.724 | 0.326 | 15 ms |
| hybrid | 0.710 | 0.868 | 0.890 | 0.754 | 0.348 | 64 ms |

Chance top-1 queue agreement is 0.169. The modes are within noise of each other, including with twins removed. Sparse scores spread out usefully (twin about 0.8, loose topical match about 0.2) so they can drive a threshold and be explained in the console; E5 cosine sits between 0.82 and 0.90 for almost everything. Sparse needs no torch at serving time (no 15 s model load, no weights download, works offline). E5 was fast enough (437 tickets/s on M3 Pro MPS, full KB in 47 s), so speed was not the reason.

### D3-11 Cross-lingual matching is rare and reported, not claimed
**Decision:** Every precedent carries its detected `answer_language`. The console and drafting step show it; we do not claim cross-lingual grounding unless the cited evidence is actually in the other language.
**Why:** Top-1 precedents differ in language from the query for only 2.3% of validation queries (sparse). The KB has German coverage for German tickets, so even dense retrieval returns German precedents for German queries (28 de→en top-1 hits out of 688 German queries, vs 21 for sparse).

### D3-12 Release retrieval mode: EmbeddingGemma, sparse as fallback (replaces D3-10)
**Decision:** Release retrieval is dense search with Google EmbeddingGemma (`embeddinggemma` via local Ollama, 768-dim, prompts `task: search result | query: ` / `title: none | text: `). If Ollama is unreachable or takes over 2 s, `Retriever.retrieve` falls back to the sparse index in the same call and `last_mode` reports `sparse`, so the API and console can show which mode served the request. A blank body returns no precedents (gate rule 1).
**Why:** The hackathon centres on Gemma open models, and EmbeddingGemma also measured best. Validation, 2,286 queries, fit-only index (`reports/retrieval_dense_embeddinggemma.json`, `reports/retrieval_no_twin.json`; the twin set is the union flagged by sparse, E5 and EmbeddingGemma, removed for every mode):

| mode | queue top-1 | no-twin queue top-1 | no-twin type | no-twin priority | de→en top-1 hits |
|---|---|---|---|---|---|
| sparse | 0.708 | 0.316 | 0.735 | 0.419 | 21 |
| E5 | 0.679 | 0.303 | 0.741 | 0.412 | 28 |
| **EmbeddingGemma** | **0.776** | **0.437** | **0.799** | **0.519** | **184** |
| RRF sparse + EmbeddingGemma | 0.724 | 0.331 | 0.756 | 0.429 | – |

On authored tickets it returns more specific precedents: the German printer ticket gets "Druckprobleme", "Drucker-Probleme" and the English "Problem with Printer Not Printing" (sparse score 0.00 for that one); the German VPN outage gets hospital IT outage tickets; the German password ticket gets English login tickets, which is real cross-lingual matching (D3-11 still applies: we report `answer_language` per precedent). Equal-weight reciprocal rank fusion with sparse is worse than EmbeddingGemma alone because sparse is much weaker on fresh tickets, so it is not used. Cost: 74 ms p50 / 79 ms p95 per query on M3 Pro, full-KB index build 5 min at ~78 tickets/s, Ollama must be running (`ollama serve`, `ollama pull embeddinggemma`, 621 MB). Fully local, so the demo works without Wi-Fi.

Score guidance for D4 (EmbeddingGemma cosine, top-1): paraphrase twins 0.85+ (queue agreement 0.97+), real support tickets with no twin about 0.45–0.55, off-topic text ("What is the weather in Paris tomorrow?", "Write me a poem about cats") about 0.19–0.21. A threshold around 0.35 separates out-of-scope from in-domain; whether the precedent actually supports the request still needs D4's check.

### D3-13 No cross-encoder reranker
**Decision:** Do not rerank EmbeddingGemma's results. `scripts/eval_rerank.py` stays so the result can be reproduced.
**Why:** Reranking EmbeddingGemma's top 20 with `BAAI/bge-reranker-v2-m3` on 500 validation queries (`reports/rerank_eval.json`):

| setting | queue top-1 | queue any-of-5 | type top-1 | priority top-1 |
|---|---|---|---|---|
| no-twin, EmbeddingGemma | 0.428 | 0.714 | 0.794 | 0.486 |
| no-twin, + rerank | 0.398 | 0.694 | 0.802 | 0.486 |
| with twins, EmbeddingGemma | 0.800 | 0.910 | 0.948 | 0.816 |
| with twins, + rerank | 0.756 | 0.896 | 0.930 | 0.792 |

It changed the top precedent for 78% of no-twin queries, made queue agreement worse, and left type and priority flat. It adds 863 ms p50 / 1,277 ms p95 per query on M3 Pro (MPS), more than ten times the retrieval itself (74 ms), which alone would break the plan's one-second local p95 target. It would also put torch and a 2.2 GB model into the serving path. Label agreement is only a proxy, so a reranker could still pick precedents a person finds closer, but nothing measured justifies the cost. The Gemma-based `bge-reranker-v2-gemma` was not tried: it is built on a 2B-class Gemma model, several times larger than `bge-reranker-v2-m3`, so it would be slower still, and the laptop has under 9 GB free.

### D3-14 `answer_snippet` carries the exact historical answer
**Decision:** The evidence store keeps each KB answer exactly as in `kb_train` (outer whitespace stripped), and `Precedent.answer_snippet` is that full text. No salutation stripping, whitespace collapsing, capitalization or truncation.
**Why:** D5's pipeline withholds any draft whose evidence span is not a substring of `answer_snippet`, D1's judge checks spans against the `kb_train` answer, and D4/D3 checks quote from the same text. With one exact text all three agree. Answers are short (median 58 words, p99 about 100), so the console can clamp the display instead.

### D3-15 Reply generation is a D3 responsibility
**Decision:** D3 owns drafting: `generation.Generator` builds the reply from the precedents and runs its own grounding checks. D4 keeps the escalation gate and PII masking.
**Why:** Team agreement during the build (the plan's ownership table and D4-03/D4-07 still list drafting under D4; D4 to update its log). Retrieval and drafting share the evidence format, the span rules and the language handling, so one owner avoids a second interface.

### D3-16 Vertex AI RAG Engine + Gemini as the main path, local stack as fallback
**Decision:** The release path is RAG Engine retrieval plus Gemini on Vertex. If either is unreachable, the same call falls back to the local stack (EmbeddingGemma retrieval, Gemma 4 drafting, extractive quotes), so the demo still works offline (T-05, D5-19). Setup: project `sentineldesk-kb-9955` on the GDG Cloud Boston x Northeastern event credit; corpus in `europe-west3` (new projects need an allowlist for `us-central1`, `us-east1`, `us-east4`, `europe-west1`); embedding model `text-multilingual-embedding-002`; one file per KB ticket named `<TICKET_ID>.txt` with the same cleaned subject + body as the local indexes; chunk size 512 so every ticket stays one chunk; imported in batches of at most 9,000 files (API limit 10,000 per request). Retrieval maps each hit's source file back to its ticket ID and takes the answer from the local evidence store. Holdout IDs are rejected on both sides.
**Why:** Team decision to use the event's Google Cloud credits visibly (D5-19), with the offline fallback the plan requires. RAG Engine scores are cosine distance; `Precedent.score` is reported as 1 − distance so that higher is better in every mode. Thresholds stay per mode.
**Status:** blocked by quota, not shipped in the 1:00 candidate. The first import (9,000 files) finished with every file failed ("internal error", code 13). Cause: the new project's quota `online_prediction_requests_per_base_model` is 5 requests per minute for every Vertex embedding model (`textembedding-gecko` family, `gemini-embedding`) in `europe-west3` and `us-central1`, and the import called the embedding model at up to 1,000 per minute (a direct call returned HTTP 429). At 5 per minute the 22,852 tickets cannot be indexed today. Only one import can run per corpus (a second returns HTTP 400), so the KB is split over three corpora, one per part (`...1856`, `...7904`, `...6976`), and `RagEngineRetriever` queries them in parallel and merges by score. Everything is scripted: after a quota increase, `scripts/rag_engine_setup.py --skip-upload` re-imports and `scripts/eval_rag_engine.py` compares it with EmbeddingGemma on the same candidates. Gemini drafting uses a separate quota and works on this project.

### D3-17 Draft format and grounding checks
**Decision:** The model returns a list of sentences, each with a `source` ticket ID and a 5–15 word `evidence_span` copied from that precedent (a greeting may have no source). D3 assembles `reply_text` with inline `[TKT-#####]` markers and the D5 `Draft` (`reply_text`, `cited_ticket_ids`, `claims`, `abstain`). `check_draft` (also exposed as `validate`) requires: every cited ID was retrieved; every span is a verbatim substring of that precedent's answer; at most two sentences without a source; numbers appear in the cited evidence or the ticket; no commitment words (refund, guarantee, credit, Erstattung, Gutschrift, ...) unless the evidence uses them; no placeholders; reply language equals the ticket language. A failing draft gets one repair attempt with the error list, then the next backend runs; the extractive fallback quotes evidence sentences verbatim. An abstaining draft is respected and not replaced by a fallback.
**Why:** The brief bans fabricated policies, prices and guarantees, and the plan's grounding contract asks for claim-level evidence. Building the markers ourselves guarantees every cited sentence links to an evidence card. Before checking: placeholders in reply sentences are filled with plain phrases ("your account number", "Ihre Kontonummer", "by phone"), a short opening greeting loses any citation, a span found verbatim in a different retrieved precedent is re-attributed to it, and a span within 0.85 similarity of an evidence substring snaps to that exact substring.

### D3-18 Generators: Gemini 3.5 Flash, then Gemma 4, then extractive
**Decision:** Primary `gemini-3.5-flash` on Vertex (global endpoint, thinking level `minimal`, JSON response schema). Fallback `gemma4:e2b-it-qat` through local Ollama (thinking off, JSON-schema constrained). Last resort: extractive quotes. 6-second budget per backend.
**Why:** `gemini-2.5-flash` returned "not found" for the new project; the publisher-model listing showed `gemini-3.5-flash` as the newest Flash model. Gemma 4 needed Ollama 0.30.5 or newer, so Ollama was upgraded from 0.15.6 to 0.40.17. Measured on 40 KB-validation tickets with fit-only retrieval (`reports/generation_eval_gemma.json`): Gemma 4 drafted 40/40 (34 by Gemma, 6 by the extractive fallback, 5 of them after hitting the 6 s budget), 82.5% of model calls passed every check on the first try, 100% of final drafts pass the checks, 100% reply-language match (27 en, 13 de), 2.2 cited claims and 46 words per draft, p50 4.0 s / p95 6.0 s. Gemini 3.5 Flash on the same 40 tickets (`reports/generation_eval_gemini.json`): 38/40 drafted by Gemini, 2 by the extractive fallback (both after the 6 s budget), 95% first-try pass rate, 100% of final drafts pass the checks, 100% reply-language match, 1.9 cited claims and 43 words per draft, p50 2.6 s / p95 5.4 s. On the authored demo tickets Gemini passed every check on the first try in 1.4–1.9 s. Lesson recorded in the code: with constrained decoding, putting `abstain` first in the output schema made Gemma abstain on everything; it now comes last.

### D3-19 Relevance guard before drafting
**Decision:** `draft_with_mode(..., min_top_score=0.35)` abstains without calling a model when the best EmbeddingGemma precedent scores below 0.35. The same 0.35 is D3's suggested starting point for D4's weak-evidence rule (rule 8) in dense mode. RAG Engine and sparse scores need their own values.
**Why:** Off-topic tickets ("What is the weather in Paris tomorrow?") score about 0.19–0.21 and real support tickets about 0.45–0.55 (D3-12). Without the guard Gemma wrote a fluent, fully quoted, irrelevant reply about maintenance windows: the checks verify quoting, not relevance.

### D3-20 One entry point for D5: `sentineldesk.d3.D3`
**Decision:** `D3.load()` wires language detection, retrieval (RAG Engine if configured, else EmbeddingGemma, else sparse) and drafting (Gemini, Gemma 4, extractive) with every fallback per request. It exposes `detect_language`, `retrieve`, `draft` and `validate` with the signatures in D5's `Components`, plus `analyze_ticket`, which returns the modes that actually served the call without shared state. The relevance guard threshold is picked per retrieval mode.
**Why:** D5 integrates one object instead of four modules, and the console can show the real retrieval and generator modes per request. With RAG Engine blocked (D3-16), the 1:00 candidate is EmbeddingGemma retrieval + Gemini 3.5 Flash drafting, with Gemma 4 and extractive fallbacks.

### D3-21 Model-judged draft quality, and two fixes it found
**Decision:** Judge drafts with `gemini-3.5-flash` (thinking `high`, temperature 0; no Pro model is available to the new project) using D1's `judge-v1` rubric verbatim for every sentence, plus a draft-level prompt (`draft-v1`): does the draft respond to the ticket, and does its next step agree with the answer the human agent actually gave (available for KB validation tickets only). `scripts/judge_generation.py`, results in `reports/judge_generation.json`. Fixes made from its findings: (1) reject "sentences" that are formatting debris (`source: TKT-11245`, ticket IDs in the text, fewer than 3 words); (2) stop filling `<tel_num>` and `<email>` generically, because they can be the customer's or ours: outside "at/unter <tel_num>" they now fail the placeholder check and go to repair.
**Why:** The deterministic checks prove quoting, not meaning. On the 40 + 40 validation drafts (before the two fixes):

| | Gemma 4 | Gemini 3.5 Flash |
|---|---|---|
| cited sentences supported | 85/88 (96.6%), 1 partial, 2 unsupported | 75/75 (100%) |
| uncited non-greeting sentences supported | 5/5 | 11/11 |
| drafts with every sentence supported | 37/40 | 40/40 |
| responds to the ticket (yes / partly / no) | 33 / 5 / 2 | 36 / 4 / 0 |
| next step agrees with the agent's real answer (yes / partly / no) | 13 / 21 / 6 | 13 / 24 / 3 |

Both "unsupported" Gemma sentences were real defects (the debris sentence, and "call `<tel_num>`" filled as "call our support phone number" when the evidence meant the customer's number). The one "not responsive" Gemma draft cited real evidence about a different problem: grounding does not guarantee relevance. After the fixes (re-run at 12:50 on the same 40 tickets, committed reports): Gemma 86/87 cited sentences supported (0 unsupported), 39/40 drafts fully supported, first-try pass 75%; Gemini 76/76 supported, 39/39 drafted fully supported (36 by Gemini, 3 by the extractive fallback, 1 abstained), first-try pass 90%, p50 2.7 s. Gemini's own 36 drafts: 34 responsive, 2 partly, 0 unrelated; the only unrelated draft in that run came from the extractive fallback, which is the weakest path (5 of its 11 drafts across both runs fully responsive) and runs only when both models fail. The stricter placeholder rule lowers first-try passes slightly (ambiguous tokens now go to repair instead of being guessed). Caveats: the judge is the same model family as the Gemini drafter (self-preference possible; the Gemma comparison shows it does flag problems); 40 drafts per model; "agrees with the agent" compares against one synthetic answer, and many "partly" cases are drafts asking for details where the agent took an action.

## Open

- Swap the provisional partition for D1's manifest once published.
- Agree the weak-evidence threshold with D4, using authored or no-twin queries (D3-09). Guidance in D3-12: EmbeddingGemma top-1 under about 0.35 means off-topic. Sparse scores use a different scale, so the fallback path needs its own threshold.
- Export the top-5 precedents for every holdout input, for D1's scorer.
- Pass the twin finding (D3-09) and the HF revision (D3-01) to D1.
- D5: serving must start Ollama with `embeddinggemma` and `gemma4:e2b-it-qat` pulled, set `GOOGLE_CLOUD_PROJECT`, and be signed in with gcloud for Gemini; otherwise the fallbacks take over.
- RAG Engine: request a quota increase for `online_prediction_requests_per_base_model` (europe-west3, textembedding-gecko), then re-import.
