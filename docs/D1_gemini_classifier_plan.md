# Gemini classifier baseline: research and execution plan (D1)

Goal: an LLM baseline for `type`, `queue`, `priority` on the 5,715 holdout tickets, scored by the same `scripts/score.py` / `run_eval.py` as the TF-IDF baseline. It answers the judge question "why not just ask an LLM?" with numbers, and gives D2 something to compare against.

Facts below were read from Google's docs on 2026-11-21 (links at the end). Prices and limits change; re-check before spending.

## 1. Research findings

### Models (Gemini API, text)

| model ID | status | standard $/1M in / out | batch $/1M in / out | notes |
|---|---|---|---|---|
| `gemini-3.5-flash-lite` | stable | 0.30 / 2.50 | 0.15 / 1.25 | cheapest; thinking defaults to `minimal` |
| `gemini-3.8-flash` | stable, newest Flash | 0.75 / 3.75 | 0.375 / 1.875 | price doubles on 2027-01-01; thinking default `medium`, lowest `low` |
| `gemini-4-carbon` | listed, no stable/preview badge | 2.00 / 10.00 | 1.00 / 5.00 | price doubles 2026-12-01 |
| `gemini-4-hydrogen` | listed, no badge | 1.50 / 7.50 | 0.75 / 3.75 | |
| `gemini-3.1-pro-preview` | preview | 2.00 / 12.00 | 1.00 / 6.00 | only Pro text model; preview models have tighter limits |

- Thinking tokens are billed as output. Set the lowest thinking level available for each model; classification does not need reasoning.
- `gemini-2.5-flash` and `gemini-2.5-pro` auto-route to newer models from 2026-12-01. Do not use them.
- Gemini 4 models have no stable badge on the models page. Confirm with a `models.list` call before relying on one.

### API features we need

- **Structured output:** JSON with a schema whose fields are string `enum`s of the exact KB labels. The model cannot return a label outside the list. Put the schema only in the request, not repeated in the prompt (Google advises this).
- **Confidence:** response-side `avgLogprobs` / `logprobsResult` fields exist, but the docs do not confirm a request flag for `generateContent` or which models support it. Plan: ask the model for a probability per label in the JSON, then calibrate it (step 5). Treat logprobs as a bonus if the smoke test shows they work.
- **Batch mode:** 50% price, JSONL up to 2 GB, structured output supported, but turnaround target is 24 h (jobs expire after 48 h). Too slow to depend on today. Use synchronous parallel calls; keep batch as a fallback for an overnight rerun.
- **Limits (Tier 1):** spend cap $10 per rolling 10 minutes, billing cap $250. RPM/TPM per model are only shown in AI Studio for the signed-in project: check there before the full run. Batch enqueued-token limit: 3M for `gemini-3.8-flash`, 10M for `gemini-3.5-flash-lite`.

### Cost of one full holdout run (5,715 tickets)

Estimate: ~1,600 input tokens per ticket with 8 retrieved examples (~750 zero-shot), ~150 output tokens including minimal thinking.

| model | few-shot, standard | few-shot, batch |
|---|---|---|
| `gemini-3.5-flash-lite` | ~$5 | ~$2.50 |
| `gemini-3.8-flash` | ~$10 | ~$5 |
| `gemini-4-carbon` | ~$27 | ~$13 |

Model selection on 300 KB validation tickets costs under $2 per model. Total budget for this plan: under $50 of credits.

### What to expect

- The labels are generator specific, not common sense. Priority follows queue (Service Outages 70% high, HR 9%), and queue names overlap ("Technical Support" vs "IT Support" vs "Product Support"). Zero-shot will likely lose to TF-IDF on queue and priority; that is still a useful, honest result.
- Few-shot with nearest KB examples (kNN) shows the model how this dataset labels similar tickets, which is where an LLM is most likely to beat or match the 0.587 / 0.631 queue / priority baseline.
- Latency is ~1 to 2 s per call, above the plan's 1 s local p95 target. This is an offline baseline and comparison, not the default serving path, unless D2 and D5 decide otherwise.

## 2. Leakage rules for this work

- Prompts contain only the ticket's `subject` + `body` from `test_inputs` and labelled examples from `kb_train`. Never `test_labels`.
- Few-shot pool: `fit` partition during model selection on `val`; full canonical KB for the holdout run. Not the `kb_extra` (20k) tickets, so the result compares cleanly with the TF-IDF baseline.
- Every retrieved example ID goes through `scripts.leakage.assert_kb_only`.
- Output file follows `docs/prediction_format.md` and is scored by `run_eval.py` in all three scopes (full, no near-copies, no twins).

## 3. Execution plan

| step | what | output | time |
|---|---|---|---|
| 0 | **You:** create an API key in AI Studio on the credits project; `setx GEMINI_API_KEY ...` locally; note the RPM/TPM shown for the candidate models | key in env, never in git | 10 min |
| 1 | Smoke test: `models.list`, one schema call per candidate model, check enum output, thinking level and whether logprobs come back | `reports/llm_smoke.md` | 15 min |
| 2 | Move the HTTP, retry and cache code out of `judge_drafts.py` into `scripts/gemini_client.py`; both scripts use it | shared client, tests still pass | 20 min |
| 3 | `scripts/llm_classify.py`: zero-shot and kNN few-shot prompts (k=8, TF-IDF neighbours from the KB pool), enum JSON schema, per-label probabilities, parallel workers, disk cache, writes a predictions file | script + unit tests with a mock backend | 45 min |
| 4 | Model selection: 300 stratified `val` tickets, pool = `fit`. Grid: {flash-lite, 3.8-flash, carbon} x {zero-shot, few-shot}. Compare macro-F1 per head, Incident+high recall, cost, p50 latency | `reports/llm_selection.md` | 30 min |
| 5 | Calibration: run the chosen config on the 2,286 `calib` tickets, fit one temperature per head on the returned probabilities | temperatures saved with the run config | 20 min |
| 6 | Full holdout run, chosen config, pool = full KB. Hash the file, score with `run_eval.py` | `reports/llm_<model>_<mode>*/` | 30 min |
| 7 | Write-up: side-by-side with TF-IDF and majority, per-language breakdown, cost, latency; decision log entries; tell D2 | README section, `D1_decisions.md` | 15 min |

Total about 3 hours of work, roughly $15 to $40 of credits. Steps 2 and 3 can start before the key exists (mock backend).

**Cut line:** if the full run is not scored by the team's 1:00 PM evaluation freeze, report the 300-ticket validation numbers as a labelled side result and keep the TF-IDF baseline as the official one.

## 4. Risks

| risk | response |
|---|---|
| 429 rate limit or $10 / 10 min spend cap | exponential backoff (already in the judge client), fewer workers; cache means a rerun only pays for missing tickets |
| Gemini 4 model not actually available on the key | fall back to `gemini-3.8-flash`; decided by the smoke test |
| Self-reported probabilities badly calibrated | temperature fit on `calib`; the scorer reports ECE honestly either way |
| Prompt injection inside a ticket | output is enum constrained, so the worst case is a wrong label; tickets are wrapped as data in the prompt |
| Cost overrun | estimate is under $50; print running token totals and stop at a set budget |

## Sources

- [Gemini models](https://ai.google.dev/gemini-api/docs/models)
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Batch mode](https://ai.google.dev/gemini-api/docs/batch-mode)
- [Structured output](https://ai.google.dev/gemini-api/docs/structured-output)
- [Thinking](https://ai.google.dev/gemini-api/docs/thinking)
- [Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- [generateContent API reference](https://ai.google.dev/api/generate-content)
- [Batch Mode launch post](https://developers.googleblog.com/en/scale-your-ai-workloads-batch-mode-gemini-api/)
