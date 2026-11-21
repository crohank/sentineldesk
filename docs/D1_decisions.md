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

## Open

- Pin the HF dataset revision and make `make_split.py` fail on a hash or count mismatch.
- Pin pandas to an exact version (3.1.1 on this machine) so the shuffle is reproducible on other machines.
- Move the drift check and baseline scoring out of `eda.py`, which currently reads `test_labels`. Only the scorer should.
