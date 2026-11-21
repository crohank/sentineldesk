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

D1 keeps its own numbered log in [docs/D1_decisions.md](docs/D1_decisions.md). This section summarizes it and lists plan items D1 hasn't logged yet.

### D1-01 to D1-12 (on `main`)
- Main CSV only (`aa_dataset-tickets-multi-lang-5-2-50-version.csv`, 28,587 rows). The brief's `load_dataset` call mixes in a file with 8,395 repeated bodies and a German file with a different taxonomy.
- Fill nulls, decode literal `\n`, dedupe on trimmed lowercased body (15 rows removed), shuffle with seed 42, assign `TKT-{i:05d}`, cut at 80%: KB 22,857 (`TKT-00000`–`TKT-22856`), holdout 5,715 (`TKT-22857`–`TKT-28571`).
- Three files: `kb_train`, `test_inputs` (ID, subject, body) and `test_labels` (scorer only). Holdout answers are never written to disk. Data stays out of git.
- Detect language from the text: the `language` column is wrong for about 26% of `de` rows. Ignore the `version` column.
- Split independently re-derived and verified. Source SHA-256 `f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0`.
- Repo stays private until submission and must be judge-accessible by 2:50.

### D1-13 to D1-16 (on branch `d1/data-eval`, not merged yet)
- D1 work goes on `d1/data-eval` and reaches `main` only through a PR.
- Near-twin paraphrases stay in the split: 520 holdout tickets (9.1%) have a KB ticket at word TF-IDF cosine ≥ 0.8. Every triage metric is reported on all 5,715 tickets and again without the near-twins. The plan's MinHash audit becomes the official twin list.
- Every metric is also broken down by language detected from the body, not by the `language` label. German tickets score clearly lower on the baseline.
- Joint Incident + high recall is reported two ways: both predictions right, and escalated on predicted high priority alone.

### From the team plan, not yet in D1's log
- **Internal KB partitions.** Split the KB (never the holdout) 80/10/10 into fit, calibration and validation, seeded and stratified by type × priority, and publish the ID manifest. All fitting, calibration and threshold tuning uses these partitions, so the holdout stays untouched until the 1:00 freeze. The manifest isn't in the repo yet, and D2 needs it before calibration.
- **Scoring rules.** A separate scorer is the only code that opens `test_labels`. It scores all 5,715 frozen predictions by ticket ID, rejecting missing, repeated or unknown IDs, after the predictions are hashed. It reports:
  - per-target macro-F1 next to the majority-class baseline
  - per-class precision, recall and F1 with support, and confusion matrices
  - recall for `type == Incident AND priority == high`, plus the share the gate sends to humans and to on-call, with counts
  - 10-bin ECE and reliability plots
  - retrieval agreement proxies
  - a generation audit on 100 holdout IDs chosen by seed before scoring
  - warm p50/p95 latency per stage

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

### D4-06 Drafts are for the operator only
**Decision:** The system never sends a customer message. An eligible ticket gets a draft for a human to review.
**Why:** Keeps a person in the loop for anything the gate misses.

### D4-07 D4 work lives on branch `D4`
**Decision:** All D4 work (PII masking, the gate, templates, drafting and validation) is committed on branch `D4` only.
**Why:** Keeps D4's changes in one place for review, the same way D1 works on its own branch (D1-13).

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

- **Python and tooling.** The plan assumes Python 3.12 with `uv` (`uv sync --frozen`, `uv run …`). The repo uses a Python 3.14 venv with `pip` and scikit-learn 1.8.0 because of Windows Smart App Control (D1-07). Pick one before the README and Dockerfile are written.
- **Pin pandas** to an exact version so the shuffle reproduces on every machine (D1 open item).
- **Holdout numbers.** `eda.py` reads `test_labels`, and D1-14 to D1-16 quote baseline scores measured on the holdout. Move label reading into the scorer, and agree that thresholds and gate rules are picked on the KB validation partition, never from holdout numbers.
- **Outage rule.** Escalate on predicted Incident + high (D4-01 rule 2), or on predicted high priority alone, which D1-16 reports catches more real high Incidents? D2 and D4 decide on KB validation.
- **Strip greeting boilerplate** ("Dear Customer Support Team, I hope this message…") before indexing and feature extraction? The EDA recommends it for retrieval.
- **Branches and merging.** D1 (`d1/data-eval`), D2 (`D2`) and D4 (`D4`) each have their own branch, and D1's work reaches `main` through a PR. Still open: the same rule for D3 and D5, and who reviews and merges PRs into `main`.
- **Repo visibility** for judges before 2:50 (D1-11).
