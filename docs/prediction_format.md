# Predictions file format (draft)

Draft for the integration owner to confirm. `scripts/score.py` reads this format.

One file per run, Parquet (`.parquet`) or JSON Lines (`.jsonl`), one row per holdout ticket. All 5,715 IDs from `data/processed/test_inputs.parquet` must appear exactly once. Missing, repeated or unknown IDs fail the run.

## Required columns

| column | type | notes |
|---|---|---|
| `ticket_id` | string | holdout ID, e.g. `TKT-00012` |
| `pred_type` | string | one of the `type` labels in `kb_train` (Incident, Request, Problem, Change) |
| `pred_queue` | string | one of the 10 `queue` labels in `kb_train`, spelled exactly |
| `pred_priority` | string | high, medium or low |

Labels are case sensitive. A value not seen in `kb_train` fails the run, so a typo can't quietly count as a wrong answer.

## Optional columns

The scorer adds a section for each one that is present.

| column | type | notes |
|---|---|---|
| `conf_type`, `conf_queue`, `conf_priority` | float in [0, 1] | confidence in the predicted label |
| `decision_status` | string | `auto_draft`, `escalate_oncall`, `human_review` or `needs_clarification` |
| `destination` | string | team or queue the ticket was handed to, free text |
| `reason_codes` | list of strings | why the decision was made |
| `retrieved_ids` | ordered list of strings | KB precedents, best first (top-k). Must all be `kb_train` IDs |
| `cited_ids` | list of strings | precedents the draft cites. Must all be `kb_train` IDs |
| `latency_ms` | float | end-to-end time for the ticket |

How the scorer reads `decision_status`:

- routed to a human: `escalate_oncall` or `human_review`
- on-call: `escalate_oncall`
- `auto_draft` and `needs_clarification` are not human routing
- other values are counted and flagged in the report, not rejected

## Example (one JSONL line)

```json
{"ticket_id": "TKT-00012", "pred_type": "Incident", "pred_queue": "Technical Support", "pred_priority": "high", "conf_type": 0.91, "decision_status": "escalate_oncall", "retrieved_ids": ["TKT-04411", "TKT-18820"], "cited_ids": ["TKT-04411"], "latency_ms": 840}
```

## Scoring

```
python scripts/score.py data/predictions/RUN.parquet --run-name RUN [--exclude-ids ids.txt]
```

Writes `reports/RUN/`. `--exclude-ids` takes a text file with one holdout ID per line and reports scores with and without those tickets.

## Draft columns (generation audit)

Read by `scripts/judge_drafts.py`, only for the 100 IDs in `reports/generation_sample_ids.txt`. Other rows may leave them empty.

| column | type | notes |
|---|---|---|
| `draft_text` | string | the reply shown to the operator; empty when there is no draft |
| `draft_abstain` | bool | true when the generator declined or the gate blocked drafting |
| `draft_claims` | list of objects | `{"text", "ticket_id", "evidence_span"}`: one per substantive claim; `evidence_span` must occur in that KB ticket's answer |

`cited_ids` (above) lists every KB ticket the draft cites. Sentences in `draft_text` not covered by a claim are graded against all cited answers.

```
python scripts/judge_drafts.py data/predictions/RUN.parquet --run-name RUN --backend gemini --model MODEL_ID
```
