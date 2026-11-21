# SentinelDesk decision log

Every decision the team takes goes here: what was decided, why, and which role owns it. Use it to answer judge questions and to settle "why did we do X" without a meeting.

**How to add one**

- Add an entry under the owning section with the next free ID (`T-` for team-wide, `D2-` to `D5-` for each role). D1 numbers its entries in its own log, so don't create `D1-` IDs here.
- Format: **Decision:** one or two sentences. **Why:** the reason or the evidence. Add **Status:** only if it isn't done yet.
- To change a decision, add a new entry that names the one it replaces, and mark the old one *Superseded by …*. Don't rewrite old entries.
- Questions that aren't settled go under [Open](#open). Move them up once decided.

Entries below are dated 2026-11-21. The first entries in each section come from the team plan. D1's detailed data log is in [docs/D1_decisions.md](docs/D1_decisions.md) and is summarized here.

| Prefix | Area |
|---|---|
| T | Team-wide: scope, timeline, contracts |
| D1 | Data and evaluation |
| D2 | Triage classifier and calibration |
| D3 | Language detection and retrieval |
| D4 | Safety gate, PII, drafting, validation |
| D5 | API, console, packaging, release |

---

## Team-wide

### T-01 MVP first, bonuses after the noon demo
**Decision:** Build the brief's four required pieces (triage, escalation gate, grounded reply, live console) end to end before any bonus feature (cross-lingual retrieval, burst detection, PII guardrail polish).
**Why:** The build window is 10:20 AM–2:00 PM. Judges test the live UI, and slide-only work isn't eligible for top prizes.

### T-02 Fixed checkpoints
**Decision:** Noon integration demo on real data · 12:20 cut line · 1:00 freeze the evaluation candidate · 1:20 cut line · 2:00 feature freeze · 2:20 fall back to the local console if hosting is unreliable · 2:50 submit.
**Why:** Leaves an hour for evaluation, rehearsal and submission problems. Each cut line already says what gets dropped, so nobody negotiates under pressure.

### T-03 Shared contracts as Pydantic models
**Decision:** `TicketInput`, `Prediction`, `Triage`, `Precedent`, `Decision`, `Claim`, `Draft` and `Validation` live in one module and are frozen by 10:40, with one fixture each for a routine draft, an on-call escalation and an insufficient-input response. Changing a field needs an entry here.
**Why:** Five people build against the same shapes in parallel, and JSON Schema/OpenAPI come for free.

### T-04 One pipeline for the console and for evaluation
**Decision:** The console, the API and batch evaluation all call the same `analyze()` function.
**Why:** The reported metrics then describe the system the judges actually use.

### T-05 Offline-first
**Decision:** The release runs on the demo laptop with no network after install. Hosted models and cloud deployment are optional add-ons with a local fallback.
**Why:** Venue Wi-Fi and API quotas are outside our control, and cloud tooling isn't installed on every laptop.

### T-06 No hardcoded demo answers
**Decision:** Demo tickets go through the real pipeline. Record the actual labels, confidences, cited IDs and backend on the release build. An unexpected result gets explained, not patched.
**Why:** Judges paste unseen tickets, so a special-cased demo would fall apart in Q&A.

---

## D1 · Data and evaluation

D1 keeps its own numbered log in [docs/D1_decisions.md](docs/D1_decisions.md). This section summarizes it.

### D1-01 to D1-12 (on `main`)
- Main CSV only (`aa_dataset-tickets-multi-lang-5-2-50-version.csv`, 28,587 rows). The brief's `load_dataset` call mixes in a file with 8,395 repeated bodies and a German file with a different taxonomy.
- Fill nulls, decode literal `\n`, dedupe on trimmed lowercased body (15 rows removed), shuffle with seed 42, assign `TKT-{i:05d}`, cut at 80%: KB 22,857 (`TKT-00000`–`TKT-22856`), holdout 5,715 (`TKT-22857`–`TKT-28571`).
- Three files: `kb_train`, `test_inputs` (ID, subject, body) and `test_labels` (scorer only). Holdout answers are never written to disk. Data stays out of git.
- Detect language from the text: the `language` column is wrong for about 26% of `de` rows. Ignore the `version` column.
- Split independently re-derived and verified. Source SHA-256 `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`.
- Repo stays private until submission and must be judge-accessible by 2:50.

### D1-13 to D1-36 (on branch `d1/data-eval`, not merged yet)
- **Process:** D1 work goes on `d1/data-eval` and reaches `main` only through a PR (D1-13). `docs/D1_handoff.md` tells each role what to use, and `reports/LEAKAGE.md` is the judge-facing leakage statement (D1-36).
- **Reproducible data:** `download_data.py` pins HF revision `ddf1c81…` and checks the source SHA-256. `make_split.py` refuses to write a split whose counts, boundary IDs or spot check differ. `requirements.txt` pins exact versions (D1-17 to D1-19).
- **KB partitions:** fit 18,285 / calib 2,286 / val 2,286, stratified on type|priority, in `data/processed/kb_partitions.parquet` (D1-20).
- **Scorer:** `score.py` is the only reader of `test_labels`. It rejects missing, repeated or unknown IDs, unseen labels, out-of-range confidences, and any holdout ID in `retrieved_ids` or `cited_ids`. "Routed to a human" means `escalate_oncall` or `human_review` (D1-22). It reports 10-bin ECE and reliability plots when confidences are present (D1-33). `run_eval.py` scores the full holdout, without the 12 near-copies and without the 520 twins in one run (D1-34). The predictions format is a draft in `docs/prediction_format.md` (D1-23).
- **Generation audit:** 100 holdout IDs are fixed in `reports/generation_sample_ids.txt` (D1-21). `judge_drafts.py` runs deterministic checks and grades every claim supported / partially supported / unsupported, with a judge model different from the drafting model (D1-29).
- **Text handling:** one shared normalizer, `scripts/text_normalize.py`, applied in code. Greeting stripping is optional and off by default. 79 placeholder spellings map to six kinds: `<name>`, `<tel_num>`, `<acc_num>`, `<email>`, `<url>`, `<amount>` (D1-30 to D1-32).
- **EDA and baselines:** `eda.py` reads only the KB; baselines are separate scripts scored by `score.py` (D1-24).
- **Twins:** 12 near-copies (MinHash) and 520 paraphrase twins (TF-IDF cosine ≥ 0.8). Every number is reported with and without each list (D1-14, D1-25, D1-26, D1-28).
- **Per-language scores:** every metric is also broken down by language detected from the body. German tickets score clearly lower (D1-15).
- **Outage metric:** joint Incident + high recall is reported two ways: both predictions right, and escalated on predicted high priority alone (D1-16).
- **Index leakage helper:** `scripts/leakage.py` `assert_kb_only(ids)` for D3's index (D1-27).

---

## D2 · Triage

### D2-01 TF-IDF features with three logistic-regression heads
**Decision:** One shared TF-IDF feature set (word 1–2 grams plus character 3–5 grams) over subject + body, with separate logistic-regression heads for type, queue and priority. Queue names come from the KB. Majority-class heads are the baseline to beat.
**Why:** Trains in seconds and is easy to inspect. Character grams help with short tickets, typos and German compounds. Fine-tuning a transformer or classifying each ticket with an LLM adds training and serving risk the window can't absorb. Dense-feature heads are a later comparison.

### D2-02 Temperature calibration per head
**Decision:** Fit one temperature per head on the calibration partition by minimizing log loss. The console shows the calibrated probability of the chosen label.
**Why:** The brief asks for a calibrated confidence. Three scalars fit quickly; isotonic calibration needs more examples in the small queues.

### D2-03 Don't bend the classifier for outage recall
**Decision:** Keep the classifier's labels as predicted. Missed outages are caught by the gate instead (D4-01 rules 2 and 6).
**Why:** Classifier recall and the extra outages caught by the gate can then be reported separately, and the console can say which one fired.

---

## D3 · Language and retrieval

### D3-01 Detect language from the text
**Decision:** An offline detector runs on subject + body and is configured to recognize other languages too, so French or Spanish tickets aren't forced into English or German. The reply language follows the detected language.
**Why:** The `language` column is unreliable (D1-08), and the brief requires detection from raw text.

### D3-02 Sparse retrieval is the release default
**Decision:** TF-IDF cosine search (SciPy) over historical subject + body, returning the top 5. A ticket's answer is looked up only after its KB ID is retrieved.
**Why:** Builds in minutes and works offline. The brief asks for 3–5 precedents.

### D3-03 Dense retrieval only if it earns it
**Decision:** Pilot `intfloat/multilingual-e5-small` (normalized embeddings, query/passage prefixes, exact NumPy inner product) on 256 KB tickets. Adopt it only if it loads by 11:10 and the projected full-KB build is under 25 minutes. Sparse and dense get separate similarity thresholds.
**Why:** E5 can match a German ticket to an English answer, which is a bonus feature. Exact search over 22,857 vectors needs no vector database.

### D3-04 The index holds KB IDs only
**Decision:** Validation runs against an index built from the fit partition only. The release index is rebuilt over the full KB. A test asserts every indexed ID is a KB ID and none is a holdout ID.
**Why:** The brief's leakage rule: holdout answers must never be searchable.

---

## D4 · Safety gate and drafting

### D4-01 Ordered escalation rules
**Decision:** Every ticket gets a reason code, an operator message and a destination. Rules apply in this order:

1. Empty or unusable body → *Needs clarification*, with no prediction or draft. A missing subject is fine.
2. Explicit widespread outage of a critical service (en/de), or predicted Incident + high → *Escalate to On-Call Engineer*. This destination wins over later rules.
3. Prompt override or fake authority ("ignore previous instructions", "ignoriere die bisherigen Anweisungen") → *Human review*, no draft.
4. Unsupported or undetermined language → *Language review*.
5. A request that needs authorization or could be destructive (refund, compensation, SLA credit, price guarantee, GDPR/DSGVO demand, delete everything) → *Specialist review*.
6. Incident and high probabilities both above the risk thresholds, even as runner-up → *On-call review*.
7. Any head below its confidence threshold → *Human review*.
8. No close-enough precedent, or the evidence doesn't cover the request → *Human review*.
9. The draft abstains, times out or fails validation → *Human review*, with precedents still shown.

**Why:** Implements the brief's gate (high + Incident, low confidence, unsupported requests) and makes every escalation explainable.
*Statuses, destinations and reason codes: D4-11. Rules 1 and 9 also run in the pipeline (D5-06 on D5's branch).*

### D4-02 How thresholds are chosen
**Decision:** Start at 0.45 per head and 0.35 for each outage probability. Search the outage pair over 0.30–0.70 in 0.05 steps and pick the highest-precision pair with at least 90% validation recall. Choose confidence and evidence thresholds on 60 hand-labelled KB-validation queries: the most draft coverage with zero unsafe drafts.
**Why:** Thresholds come from data the holdout never touches. Sixty examples support an operating choice, not a safety guarantee.

### D4-03 Offline template drafts by default, Gemma optional
**Decision:** The default generator fills reviewed English/German action templates tied to evidence spans in the retrieved answers. Gemma 3 4B (instruction-tuned) is switched on only if it passes German quality, JSON parsing and latency checks on validation. It gets a 6-second budget and one repair attempt, then falls back to the offline path. The console shows which generator produced each draft.
**Why:** The template path can't invent facts and always works. An LLM writes better prose but needs checking and can fail at the venue.

### D4-04 Draft validation
**Decision:** Every cited ID must be one of the retrieved precedents, every quoted span must appear in that ticket's answer, and every substantive sentence needs support. Numbers, currencies, durations, scope, negation and qualifiers are compared within the claim, so "up to 48 hours" never supports "guaranteed within 48 hours".
**Why:** The brief bans fabricated policies, prices and guarantees. Historical answers almost never mention money, percentages, refunds or SLA hours (under 0.5%), so such a claim in a draft is almost always unsupported.

### D4-05 PII and placeholders
**Decision:** Regex-mask emails, phone numbers, IP addresses and labelled account IDs in incoming tickets and answer excerpts before logging, display or any external model call. Keep ticket IDs and error codes. Mask tokens in historical answers (`<tel_num>`, `<name>`, `<acc_num>`, `[Name]` and similar) are never copied into a draft as-is.
**Why:** Bonus PII guardrail, and the answers are full of placeholders that would look broken in a reply.
*Details: D4-13.*

### D4-06 Drafts are for the operator only
**Decision:** The system never sends a customer message. An eligible ticket gets a draft for a human to review.
**Why:** Keeps a person in the loop for anything the gate misses.

### D4-07 D4 work lives on branch `D4`
**Decision:** All D4 work (PII masking, the gate, templates, drafting and validation) is committed on branch `D4` only.
**Why:** Keeps D4's changes in one place for review, the same way D1 works on its own branch (D1-13).

### D4-08 Files D4 owns
**Decision:** D4 adds only `sentineldesk/safety/`, `tests/test_safety_*.py` with `tests/safety_fixtures.py`, `scripts/d4/`, `reports/d4/` and `docs/D4_demo_tickets.md`, plus entries in this file. It does not add `sentineldesk/__init__.py` (D2 and D5 each create one) or `tests/conftest.py` (D1 has one), and it doesn't edit `requirements.txt`, `README.md` or `.gitignore`. Tests run with `python -m pytest` from the repo root.
**Why:** No path overlaps another branch, so D4's PR merges cleanly in any order.

### D4-09 D4 follows the shared contract
**Decision:** `sentineldesk/safety/contracts.py` mirrors `sentineldesk/schemas.py` and `MaskedTicket` from D5's branch: the same field names and the same status, destination and reason-code values, as plain dataclasses. D4 reads its inputs by attribute, so D2's and D3's objects work unchanged. Once D5's contract is merged, `contracts.py` re-exports it instead. D4's stages match `Components`: `mask=mask_ticket`, `gate=gate`, `draft=draft`, `validate=validate`.
**Why:** The contract exists but isn't merged. Mirroring it keeps D4 independent and makes the switch a one-file change. Dataclasses add no dependency.

### D4-10 Local environment and data
**Decision:** `.venv` (Python 3.14) installs D1's exact pins. The data is built in a separate worktree of `d1/data-eval` with D1's scripts, and only `kb_train`, `kb_partitions` and `test_inputs` are copied into `data/processed/`. Counts and hashes match D1-12 and D1-20.
**Why:** D4 stays independent of D1's unmerged branch, and holdout labels are never on disk in the D4 checkout.

### D4-11 Statuses, destinations and reason codes confirmed
**Decision:** D4 confirms the contract's values (D5-04 on D5's branch):

| rule | status | destination | reason code |
|---|---|---|---|
| 1 | needs_clarification | requester | INSUFFICIENT_INPUT |
| 2 | escalate | on_call | HIGH_IMPACT_OUTAGE (wording) or PREDICTED_INCIDENT_HIGH (classifier) |
| 3 | review | human_review | INSTRUCTION_OVERRIDE |
| 4 | review | language_review | UNSUPPORTED_LANGUAGE |
| 5 | review | specialist_review | AUTHORIZATION_REQUIRED |
| 6 | escalate | on_call | OUTAGE_RISK |
| 7 | review | human_review | LOW_CONFIDENCE, one reason per uncertain head |
| 8 | review | human_review | WEAK_EVIDENCE: no close precedent, no usable answer, or an integration request |
| 9 | review | human_review | DRAFT_FAILED |
| none | draft_ready | operator | none |

On-call wins over review, and the reason of every rule that fired is kept. When several review rules fire, the destination follows the first of them in rule order. For the scorer: draft_ready = auto_draft, escalate = escalate_oncall, review = human_review (`SCORER_STATUS` in `contracts.py`). Console badges: Auto-Draft Ready, Escalate to On-Call Engineer, Human Review, Needs Clarification.
**Why:** One closed set of values for the console, the logs and the scorer, with the scorer's different names mapped in one place. The gate repeats rule 1 with the pipeline's test (any letter), so `gate()` is safe to call on its own.

### D4-12 Gate patterns in English and German
**Decision:** `patterns.py` holds the wording rules. Rule 2 fires only when an outage term, a broad-scope term and a critical-service term all appear. Rule 3 uses high-precision override phrases. Rule 5 needs a commitment term and a demand cue in the same sentence, or a strong phrase such as "full refund", "erstatten Sie mir" or "delete all". Integration and compatibility requests always go to review under rule 8. On the KB validation partition (2,286 tickets) they fire on 0.1%, 0%, 0.3% and 12.5% of tickets.
**Why:** The wording rules are precise by design; outage recall comes from the classifier part of rule 2 and from rule 6. Drafts never confirm integrations, and the brief names invented integrations as a failure, so integration requests go to a person at the cost of about one draft in eight.

### D4-13 PII masking
**Decision:** `mask_pii` masks emails, phone numbers (8–15 digits), IPv4 and IPv6 addresses, IBANs, card numbers (Luhn-checked) and labelled account IDs, using the KB's placeholder kinds: `<email>`, `<tel_num>`, `<acc_num>`, plus `<ip_address>`. Ticket IDs, error codes, versions and dates are kept. `mask_ticket` is the pipeline's mask stage, and the same function masks answer excerpts before display and any text sent to a hosted model.
**Why:** Masked tickets then use the same tokens as the training data (D1-32). Names are not masked; that limit is stated rather than hidden.

---

## D5 · Integration and release

### D5-01 One FastAPI process
**Decision:** One Python package. FastAPI serves `POST /api/triage`, `GET /healthz` and a server-rendered console with local assets. The response carries the request ID, detected language, triage, decision, precedents, validated draft (nullable), per-stage and total milliseconds, release ID, retrieval mode and generator mode.
**Why:** One start command, one container, and a stable JSON API judges can call. A separate frontend is integration work we don't need.

### D5-02 Logs without ticket text
**Decision:** JSON logs carry the request ID, release hash, predicted classes, reason codes, model mode and stage timings. No ticket text, answers or prompts.
**Why:** PII hygiene, while still enough to debug and measure latency.

### D5-03 Serving bundle contents
**Decision:** The bundle holds model artifacts, the index and redacted KB evidence only: no raw CSV, holdout inputs, holdout labels or secrets. Everything loads at startup; a request never trains or downloads anything.
**Why:** Leakage rule and predictable latency.

### D5-04 Burst detection stays off the critical path
**Decision:** If built: a 10-minute in-process window where three distinct submissions about the same service, with outage language and similar vectors, raise a "Major Incident" banner. Disabled during scoring; the single-instance limitation is disclosed.
**Why:** Bonus feature, started only after the noon demo works.

---

## Open

- **Python and tooling.** The plan assumes `uv` with Python 3.12. D1 pins exact versions for `pip` (D1-19), and D5's branch adds `pyproject.toml` and `uv.lock` (`requires-python >=3.12,<3.15`). Settle one requirements source before the README and Dockerfile are final.
- **Holdout numbers.** D1 reports holdout baselines for reference only (D1-24). Thresholds and gate rules are picked on the KB validation partition, never from holdout numbers.
- **D2's split.** D2's branch trains on its own 80/10/10 split built from all three CSVs (`scripts/load_split_811.py`), not on D1's KB partitions. That split can put D1's holdout tickets into D2's training data. Train on D1's fit partition before any holdout scoring.
- **Two decision files.** D5's branch adds `decisions.md` at the repo root, and this file is `DECISIONS.md`. On macOS and Windows those are the same file name, so merging both breaks checkouts. Rename one before either reaches `main`. D5's own log also numbers D5-01 to D5-21 differently from the plan-derived D5 entries here; D5's numbering wins and the D5 section here becomes a summary.
- **Status names.** D5's contract uses `draft_ready`, `escalate`, `review` and `needs_clarification`. D1's scorer reads `auto_draft`, `escalate_oncall`, `human_review` and `needs_clarification`. The predictions writer maps draft_ready → auto_draft, escalate → escalate_oncall, review → human_review, or one side renames.
- **Predictions format.** D5 to confirm `docs/prediction_format.md` (D1-23), including the draft columns.
- **Outage rule.** Escalate on predicted Incident + high (D4-01 rule 2), or on predicted high priority alone, which D1-16 reports catches more real high Incidents? D2 and D4 decide on KB validation.
- **Password-reset demo.** All 483 KB password tickets are labelled Incident or Problem, none Request, so the planned password-reset demo has no routine precedent. D4 and D5 pick another routine ticket.
- **Threshold-check labels.** The plan has D3 and D4 hand-label 60 KB-validation queries for whether the retrieved answer safely covers the request. D1's log doesn't include this. Confirm who labels.
- **Judge model.** The generation judge must differ from the drafting model (D1-29). Pick both on the team's GCP project.
- **Greeting stripping.** Off by default (D1-31). Measure it on D3's dense embeddings once they exist.
- **Branches and merging.** D1 (`d1/data-eval`), D2 (`D2`), D4 (`D4`) and D5 (`ajin-frank-justin`) each have their own branch, and D1's work reaches `main` through a PR. Still open: D3's branch, and who reviews and merges PRs into `main`.
- **Repo visibility** for judges before 2:50 (D1-11).
