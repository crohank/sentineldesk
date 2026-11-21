# D1 decision log (data and evaluation)

One entry per decision. Newest at the bottom. Each says what was decided and why, so a judge question or a teammate can be answered from here.

## 2026-11-21

### D1-01 Use only the main CSV
**Decision:** Build everything from `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (28,587 rows). Ignore the other two files in the HF repo.
**Why:** `load_dataset(...)` from the brief concatenates all three files (61,765 rows, the "~61,800" in the brief). The 20k file repeats 8,395 bodies from the main file, so a random split over the concatenation leaks test tickets into the KB. The German file has a different taxonomy (42 queues, 5 priorities) and no `type` or `answer`.

### D1-02 Cleaning order
**Decision:** Fill null `subject` and `answer` with `""`, then replace literal `\n` with real newlines in subject, body and answer, then deduplicate.
**Why:** 13.5% of subjects are null and the pipeline must not crash on them. About 4% of bodies and answers contain the two characters `\n` instead of a line break. Decoding before dedup means two tickets that differ only in escaping count as duplicates.

### D1-03 Deduplicate on trimmed, lowercased body, keep first source occurrence
**Decision:** Drop rows whose `body.strip().lower()` was already seen, in source file order, before shuffling. Removes 15 rows, leaving 28,572.
**Why:** Without it, one copy could land in the KB and the other in the holdout. Exact `body` match finds 0 duplicates; whitespace and case differences hide the 15.

### D1-04 Split recipe follows the brief
**Decision:** `sample(frac=1.0, random_state=42)`, reset index, assign `TKT-{i:05d}`, KB = first `int(0.8 * n)` rows. Result: KB 22,857 (`TKT-00000`..`TKT-22856`), holdout 5,715 (`TKT-22857`..`TKT-28571`).
**Why:** Same recipe as the challenge's quick-start, so judges recognize it. Not stratified; label shares still differ by at most 1.3 points between the two sides.

### D1-05 Three output files, holdout answers never saved
**Decision:** `kb_train.parquet` (all columns), `test_inputs.parquet` (`ticket_id`, `subject`, `body`), `test_labels.parquet` (`ticket_id`, `type`, `queue`, `priority`, `language`). Only the scorer reads `test_labels`.
**Why:** Inference code physically cannot see labels or answers if it is only handed `test_inputs`. Holdout answers are not on disk outside the raw CSV, so they cannot reach the index.

### D1-06 Data stays out of git
**Decision:** `data/raw/` and `data/processed/` are gitignored. Everyone regenerates with `download_data.py` then `make_split.py`.
**Why:** CC-BY-NC dataset, about 70 MB raw, and the split is deterministic (rerun gives byte-identical parquet files).

### D1-07 Python 3.14 venv, scikit-learn pinned to 1.8.0
**Decision:** Use the Python 3.14 install, not the uv-managed 3.11.
**Why:** Windows Smart App Control on this laptop blocked the uv Python 3.11 executable and the scikit-learn 1.9.1 `_loss` DLL. 3.14 plus scikit-learn 1.8.0 loads.

### D1-08 Do not trust the `language` column
**Decision:** Recommend to the team that language is detected from the ticket text, and the reply language follows the detected language.
**Why:** In a 3,000-row langdetect sample, about 26% of rows labelled `de` have an English body (0.5% the other way). The brief also requires detection from raw text at demo time.

### D1-09 Ignore the `version` column
**Decision:** Not a feature and not a target.
**Why:** It records the generator round (400 / 52 / 51), not anything about the ticket. All versions share the same label sets and a similar priority mix, and live tickets will not have it.

### D1-10 Shared greeting openings are not leakage (preliminary)
**Decision:** Treat the 511 holdout bodies that share their first 80 characters with a KB ticket as template text, not duplicates.
**Why:** They are boilerplate greetings ("Dear Customer Support Team, I hope this message..."), and labels agree on only 41% of those pairs for queue. The MinHash near-duplicate audit in the plan will replace this check.

### D1-11 Repo private for now
**Decision:** `crohank/sentineldesk` stays private. Teammates get collaborator access.
**Why:** Team's call. The brief requires public or judge-accessible by submission, so change visibility or add judges before 2:50 PM.

### D1-12 Split verified independently
**Decision:** Accept the split as canonical.
**Why:** A separate re-derivation from the raw CSV (not importing `make_split.py`) matched every row and passed 22 checks: counts, boundary IDs, exact column sets, disjoint IDs, zero cross-split duplicate bodies, unchanged labels, `TKT-09746` spot check. Source CSV SHA-256 `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`.

### D1-13 D1 work on its own branch
**Decision:** All D1 work goes on branch `d1/data-eval`. Nothing more is pushed straight to `main`; it lands through a PR.
**Why:** Five people share the repo, and `main` should only get reviewed changes. The first two commits (repo setup, this log) were already on `main` before this rule.

### D1-14 Near-twin paraphrases stay in the split, scores reported both ways
**Decision:** Keep the canonical split. The scorer reports every triage metric on all 5,715 holdout tickets and again with near-twins excluded.
**Why:** 520 holdout tickets (9.1%) have a KB ticket at word TF-IDF cosine >= 0.8, and labels agree on 99.8 to 100% of those pairs. They are generator paraphrases ("seeking guidance on integrating Steinberg Cubase 11..." vs "seeking detailed guidance on integrating..."), not exact duplicates, so D1-03 does not catch them. They inflate scores: queue macro-F1 0.587 overall vs 0.555 without them. This refines D1-10, which only looked at shared openings. The MinHash audit from the plan becomes the official twin list; the cosine check is the cross-check.

### D1-15 Score per detected language
**Decision:** The scorer breaks every metric down by language detected from the body (langdetect), not by the `language` label. Report both breakdowns in the README if space allows.
**Why:** The label is wrong for 3,406 of 28,572 rows (11.9%; 3,353 labelled `de` but English, 53 the other way). German is much weaker on the baseline: queue macro-F1 0.455 for detected German vs 0.643 for English, priority 0.518 vs 0.682. Averaging hides that.

### D1-16 Outage metric reports both escalation definitions
**Decision:** Report joint Incident+high recall two ways: both predictions right (baseline 0.654) and escalated on predicted high priority alone (baseline 0.735).
**Why:** The gate can escalate on high priority without needing the type prediction, and the gap is 8 points. D2 picks the operating rule; D1 reports both with counts.

### D1-17 Source pinned to an HF revision and a file hash
**Decision:** `download_data.py` downloads revision `ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb` and exits non-zero if the main CSV's SHA-256 is not `f187c090...6bc0`. Only the main CSV is hashed; the other files are pinned by revision only.
**Why:** An upstream edit would still parse and split cleanly, just into different ticket IDs. Only the main CSV feeds the split.

### D1-18 make_split.py refuses to write a different split
**Decision:** Before writing, it checks the source hash (before reading), then row counts, duplicates removed, boundary IDs and the `TKT-09746` spot check. Any mismatch raises `SplitMismatch` listing every expected vs observed value, and the output folder is not created. Output parquet hashes are not checked. Added `--out` to both data scripts.
**Why:** Plan rule: never silently produce a new ID mapping. Parquet bytes depend on the pyarrow version, while input hash plus counts and IDs already lock the mapping. Tested with three deliberate mismatches.

### D1-19 Exact version pins
**Decision:** `requirements.txt` pins exact versions from the working venv: pandas 3.1.1, pyarrow 26.0.1, huggingface_hub 2.4.0, langdetect 1.0.9, matplotlib 3.11.2, scipy 1.18.1, scikit-learn 1.8.0, pytest 9.2.0.
**Why:** The shuffle is only guaranteed identical on the same pandas version.

### D1-20 KB partitions: 80/10/10 stratified on type|priority
**Decision:** `make_partitions.py` sorts KB by `ticket_id`, then two `train_test_split` calls (0.2, then 0.5 of that), both seed 42, both stratified on `type|priority`. fit 18,285, calib 2,286, val 2,286. Output `data/processed/kb_partitions.parquet` (SHA-256 `7e70ffd4...`), summary in `reports/kb_partitions_summary.md`.
**Why:** Plan requirement. Sorting first makes it independent of file row order. Type and priority shares differ by under 0.1 points across partitions; queue up to 2.2 and language 1.3, since those were not stratified.

### D1-21 Generation audit sample fixed now
**Decision:** 100 holdout IDs drawn with seed 42 from the sorted `test_inputs` IDs, saved in `reports/generation_sample_ids.txt`.
**Why:** Plan requirement: chosen before any scoring so nobody can pick favourable tickets.

### D1-22 Scorer rules
**Decision:** `score.py` is the only code that reads `test_labels`. It hashes the predictions file first, and rejects missing, repeated or unknown IDs, labels never seen in kb_train, nulls, confidences outside [0, 1], and any holdout ID in `retrieved_ids` or `cited_ids`. "Routed to a human" means `escalate_oncall` or `human_review`; `needs_clarification` does not count. Per-language macro-F1 averages over classes present in that slice. Language is detected with langdetect seed 0. A training-majority reference is computed in every report.
**Why:** Bad inputs should fail loudly instead of quietly lowering scores. A holdout ID in retrieval output is leakage by definition.

### D1-23 Predictions file format (draft)
**Decision:** One row per holdout ticket, required `ticket_id`, `pred_type`, `pred_queue`, `pred_priority`; optional confidences, decision fields, `retrieved_ids`, `cited_ids`, `latency_ms`. Spec in `docs/prediction_format.md`.
**Why:** The scorer needs a fixed contract. Still a draft until D5 confirms the live pipeline can emit it.

### D1-24 EDA reads only the KB; baselines are separate scripts
**Decision:** `eda.py` no longer opens `test_inputs` or `test_labels`. The TF-IDF baseline moved to `baseline_tfidf.py`, plus a new `baseline_majority.py`; both write prediction files and are scored by `score.py`.
**Why:** Plan rule that only the scorer opens labels. Scored: TF-IDF macro-F1 type 0.884, queue 0.587, priority 0.631; majority 0.144, 0.046, 0.191; Incident+high recall 0.654 (685/1047) vs 0.

### D1-25 Near-duplicate audit method
**Decision:** MinHash in numpy (not datasketch), word 5-grams on lowercased, whitespace-collapsed bodies, 128 hashes in 16 bands of 8, seed 42, every candidate verified by exact Jaccard >= 0.85. Bodies under 5 words become one shingle. An exact all-pairs scan cross-checks recall. Label agreement uses KB-KB pairs only.
**Why:** No new dependency, so no Smart App Control risk. Exact scan found the same 12 holdout tickets, so MinHash missed none. The audit must not read test labels.

### D1-26 Two twin lists, different meanings
**Decision:** The plan's MinHash audit flags 12 holdout tickets (near-copies). The TF-IDF cosine >= 0.8 check from D1-14 flags 520 (paraphrases). Both are reported; neither replaces the other.
**Why:** 5-gram Jaccard misses paraphrases where a word changes every few words, and those are what inflate scores (queue 0.587 to 0.555 without the 520). Excluding only the 12 would understate the inflation.

### D1-27 Leakage helper for the index
**Decision:** `scripts/leakage.py` exposes `assert_kb_only(ids)`, raising `LeakageError` (an `AssertionError` subclass) with counts and example IDs if any ID is a holdout ID or not in kb_train. D3 calls it on the index IDs.
**Why:** Plan acceptance test "KB-only index membership". A subclass survives `python -O`, a bare `assert` does not.

### D1-28 Paraphrase-twin list saved for the scorer
**Decision:** `scripts/twin_audit.py` flags holdout tickets whose subject + body has word 1-2 gram TF-IDF (min_df=2, sublinear tf, fit on KB) cosine >= 0.8 to any KB ticket. Writes `reports/twin_holdout_ids.txt` (520 IDs) and `reports/twin_audit.md`. Reads only kb_train and test_inputs. Scored as `reports/tfidf_logreg_no_twins/`.
**Why:** Reproduces the D1-14 count exactly (520, and 92 at >= 0.9). Unigram body-only TF-IDF flags 1,430 at the same threshold because single shared words (product names, boilerplate) score high, so the setting matters and is fixed here. Baseline without twins: macro-F1 type 0.876, queue 0.555, priority 0.602; Incident+high recall 0.636 (615/967).

### D1-29 Generation judge
**Decision:** `scripts/judge_drafts.py` grades the 100 fixed sample drafts. Deterministic checks per draft: citations are KB IDs that were retrieved, claim IDs are among the cited, evidence spans occur in the cited answer, numbers not in the cited answers, mask placeholders, reply language vs ticket language. A judge grades every claim supported / partially_supported / unsupported against the full cited answers. Draft sentences not listed as claims are graded too. Backends: `gemini` (API key), `vertex` (project + token), `manual` (CSV sheet), `mock` (offline rules, for tests only). Prompt version `judge-v1`; judge ID, prompt hash and predictions hash go in every report. Responses are cached in `data/judge_cache/`. A holdout ID cited as evidence fails the run.
**Why:** Plan requires a model judge with recorded ID and prompt version, or the same rubric by hand. Plain HTTP (urllib) instead of the Google SDK keeps it free of new compiled dependencies under Smart App Control. `--model` has no default: pass a different model from the one that wrote the drafts, so the judge does not grade its own output. Denominators are emitted drafts, per the plan.

### D1-30 One shared text normalizer, applied in code, not in the split
**Decision:** `scripts/text_normalize.py` provides `normalize_text`, `ticket_text` and `find_placeholders`. Always on: HTML tags (known tag names only, so `<tel_num>` survives), whitespace, NFC. Optional: `strip_greeting` (EN/DE salutations, pleasantries, sign-offs) and `placeholders` = keep / canonical / remove. The parquet files stay unnormalized.
**Why:** Live tickets arrive raw, so training, retrieval and serving must run the same function, or the model trains on clean text and serves messy text. Pattern lists were built from the most common first and last sentences in the data. "I am writing to inquire" is kept because it signals request vs incident.

### D1-31 Greeting stripping is optional, not default
**Decision:** `strip_greeting` defaults to False. It never returns an empty body (falls back to unstripped text; 2 KB bodies hit this).
**Why:** Measured on sparse TF-IDF it changes nothing: macro-F1 moves under 0.004 per head, top-1 neighbour queue agreement 0.707 both ways, because IDF already discounts boilerplate. It touches 37% of KB bodies (6.8 words each), so it may still help dense embeddings, LLM prompts and UI snippets; that is unmeasured.

### D1-32 Placeholders mapped to six kinds
**Decision:** 79 spellings (20,681 occurrences) map to `<name>`, `<tel_num>`, `<acc_num>`, `<email>`, `<url>`, `<amount>`. Unknown tokens like `[Company]` are left alone. `find_placeholders` returns sorted kinds for D4's draft validator.
**Why:** Mixed EN/DE spellings (`[Telefonnummer]`, `<Tel-Nummer>`) would otherwise each need their own check.

### D1-33 Calibration measured by the scorer
**Decision:** When `conf_<head>` columns exist, `score.py` reports 10-bin top-label ECE (right-closed bins), a per-bin table and a reliability plot per head, overall and per detected language. Baseline ECE: type 0.026, queue 0.081, priority 0.018.
**Why:** Plan requirement, and it lets D2 just emit calibrated confidences. Queue is underconfident (mean confidence 0.50 vs accuracy 0.58), and priority's low overall ECE hides en 0.040 vs de 0.082.

### D1-34 One command scores all three scopes
**Decision:** `run_eval.py` scores full, without the 12 near-copies, and without the 520 twins, and writes `reports/<run>/summary.md` with the predictions hash and commit. It stops if the predictions file changes between scopes.
**Why:** D1-14 and D1-26 require every number both ways; one command keeps the three from drifting apart.

### D1-35 Report commit stamp ignores reports/
**Decision:** `-dirty` is added to the stamped commit only when tracked files outside `reports/` have uncommitted changes.
**Why:** Regenerating tracked reports marked every run dirty, so the stamp could never be clean.

### D1-36 Leakage statement and team handoff
**Decision:** `reports/LEAKAGE.md` is the judge-facing leakage statement; `docs/D1_handoff.md` tells each role what to use. LEAKAGE.md says plainly that the index check is not wired in yet.
**Why:** Judge question 1 in the plan. Claims about other people's code stay as requirements until that code exists.

### D1-37 Do not add the 20k file to the KB
**Decision:** Keep the KB at the 22,857 main-file tickets. `scripts/audit_20k.py` and `reports/audit_20k.md` stay as the record; `baseline_tfidf.py --extra` reproduces the experiment.
**Why:** Of the 20k file's 19,998 non-empty rows, 6,685 repeat a KB body and 1,717 repeat a holdout body (merging the files and re-splitting would have leaked those). 11,595 are new and 11,489 of those are not holdout twins; labels agree with the main file on 99.7% or more of the repeats, so the data is consistent. But training TF-IDF on KB + 11,489 made every head slightly worse on the unchanged holdout: macro-F1 type 0.884 to 0.879, queue 0.587 to 0.569, priority 0.631 to 0.624 (same direction in the no-twins scope; `reports/tfidf_logreg_plus20k/summary.md`). Incident+high recall on predicted high alone rose only 0.735 to 0.741. More rows from an older generator version do not pay for the added risk.

### D1-38 Adopt the 20k file as an opt-in KB extension (supersedes D1-37)
**Decision:** The 11,489 approved 20k-file rows become `data/processed/kb_extra.parquet` (IDs `EXT-00000`.., kb_train columns minus `version`, plus `source`). `audit_20k.py` refuses to write unless the file's SHA-256 and every count match. `leakage.kb_ids` / `assert_kb_only` and `score.py` accept EXT IDs as KB precedents; holdout IDs are still rejected. Label sets and the majority reference stay kb_train only. EXT rows go to training (`fit`) and the retrieval index only; `calib` and `val` stay main-file only, so thresholds are tuned on the holdout's distribution. The holdout is unchanged.
**Why:** Team call (user). The data is clean: labels agree with the main file on 99.7% or more of shared tickets, every leaked or twin row is removed, and it adds 11,489 answered precedents for retrieval. The TF-IDF drop in D1-37 (queue 0.587 to 0.569) is one linear model; larger models may use the extra rows better. D2 should confirm on `val` with their real model before shipping, not on the holdout.

### D1-39 Do not use the German file
**Decision:** The 13k German file stays out. `scripts/audit_german.py`, `scripts/experiment_german_priority.py` and `reports/audit_german.md` record the test.
**Why:** It is clean and genuinely German (2,000/2,000 detected `de`, no overlap or twins with the holdout; nearest holdout cosine 0.289), but retraining the priority head with it did nothing useful. German priority macro-F1 rose 0.002 to 0.004, English fell 0.003 to 0.005, German priority ECE got worse in every variant (0.082 to 0.090 to 0.096), across all three mappings tried (critical/very_low folded, dropped, folded at weight 0.5). Its topics (groceries, books, gardens) are far from IT support, and the 5-to-3 priority mapping is an untested assumption.

## Open

- D2: compare their model trained with and without `kb_extra` on the `val` partition; drop the extension if it does not help (D1-38).

- D3 to call `assert_kb_only` on the index IDs, then update LEAKAGE.md.
- Measure `strip_greeting` on the dense embedding model once D3 has one.
- Judge model ID for the real run: pick one available on the team's GCP project, different from D4's drafting model.
- D5 to confirm the predictions format (D1-23), including the draft columns.
- Tell D4/D5: no KB password ticket is a Request (all 483 are Incident or Problem), so the planned password-reset demo has no routine precedent. Pick another routine demo ticket.
