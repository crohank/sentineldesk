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

## Open

- Swap the provisional partition for D1's manifest once published.
- Agree the weak-evidence similarity threshold with D4, using authored or no-twin queries (see D3-09). Starting guidance for sparse: 0.5 or more is a near-identical precedent, 0.2–0.5 topical, under 0.15 weak.
- Pass the twin finding (D3-09) and HF revision (D3-01) to D1.
