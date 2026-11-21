# D1 handoff (data and evaluation)

Branch `d1/data-eval`. Decisions with reasons: [D1_decisions.md](D1_decisions.md). Leakage statement: [../reports/LEAKAGE.md](../reports/LEAKAGE.md).

## Everyone

```bash
pip install -r requirements.txt
python scripts/download_data.py    # pinned revision, SHA-256 checked
python scripts/make_split.py       # fails loudly if anything differs
python scripts/make_partitions.py
python -m pytest -q
```

- Only use the main CSV. `load_dataset(...)` from the brief pulls in duplicates and an incompatible German file.
- Never open `data/processed/test_labels.parquet`. Only `scripts/score.py` does.
- Do not trust the `language` column (about 12% wrong). Detect language from the text.
- Build every model input with `scripts.text_normalize.ticket_text(subject, body, ...)` so training, retrieval and the live API see the same text.

## D2 (triage)

- Fit on `partition == "fit"`, calibrate on `"calib"`, pick thresholds on `"val"`: `data/processed/kb_partitions.parquet` (fit 18,285 / calib 2,286 / val 2,286, stratified on type|priority).
- Baseline to beat (TF-IDF + LogReg, full holdout): macro-F1 type 0.884, queue 0.587, priority 0.631. Without the 520 paraphrase twins: 0.876 / 0.555 / 0.602.
- Incident+high recall: 0.654 (685/1047) when both heads must say Incident and high, 0.735 (770/1047) on predicted high alone, at precision 0.632 vs 0.342. The scorer reports both; you pick the gate rule.
- German is the weak spot: queue 0.455 de vs 0.643 en, priority 0.518 vs 0.682.
- Priority follows queue, not type (Service Outages 70% high, HR 9%).
- The scorer computes 10-bin ECE and reliability plots from `conf_<head>` columns, so you only need to output calibrated confidences.

## D3 (retrieval)

- Index `kb_train` only. Before shipping, run `from scripts.leakage import assert_kb_only; assert_kb_only(index_ids)`.
- `strip_greeting=True` removes salutations, pleasantries and sign-offs (touches 37% of KB bodies, 6.8 words per touched body). On sparse TF-IDF it changed nothing measurable (top-1 neighbour queue agreement 0.707 both ways), because IDF already discounts boilerplate. Expect it to matter for dense embeddings and LLM prompts; measure before relying on it. See `reports/normalization_summary.md`.
- Output `retrieved_ids` in rank order. The scorer rejects any holdout ID there and reports top-1 / any-of-top-5 queue and type agreement.

## D4 (safety and drafting)

- Answers contain mask tokens: 79 spellings, 20,681 occurrences (`<tel_num>`, `<name>`, `<acc_num>`, `[Name]`, `[Telefonnummer]`, ...). `normalize_text(..., placeholders="canonical")` maps them to six kinds; `find_placeholders(draft)` returns the kinds present, so a non-empty result rejects the draft.
- Answers almost never mention money, percentages, refunds, guarantees or SLA hours (all under 0.5%). Any of these in a draft has no precedent.
- 100 holdout IDs for the generation audit are fixed in `reports/generation_sample_ids.txt`. Do not choose others.
- Demo risk: all 483 KB tickets mentioning password/Passwort are Incident or Problem, none are Request. The planned routine password-reset demo has no routine precedent. Pick another routine ticket. "SLA" appears in 0 KB bodies (good for the refund/SLA escalation demo). "VPN" appears in 72 (outage demo has precedents).

## D5 (integration)

- Predictions format for evaluation: [prediction_format.md](prediction_format.md). Needs your confirmation that the pipeline can emit it.
- One command scores a run in all three scopes (full, no near-copies, no twins): `python scripts/run_eval.py data/predictions/<run>.parquet --run-name <run>`.
- Evaluation freeze is 1:00 PM. Send D1 the frozen predictions file; D1 hashes it before scoring.
