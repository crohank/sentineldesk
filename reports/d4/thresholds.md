# Thresholds (KB validation partition, provisional stand-ins)

## Evidence (rule 8)

Lowest best-precedent similarity where the precedent's queue and type both match the ticket for at least 80.0% of tickets at or above it.

| similarity | agreement at or above | share at or above |
|---|---|---|
| 0.10 | 65.8% | 100.0% |
| 0.12 | 65.8% | 100.0% |
| 0.14 | 65.9% | 99.9% |
| 0.16 | 66.2% | 99.3% |
| 0.18 | 66.8% | 97.9% |
| 0.20 | 67.9% | 95.6% |
| 0.22 | 70.6% | 91.2% |
| 0.24 | 73.5% | 86.0% |
| 0.26 | 76.7% | 80.7% |
| 0.28 | 80.2% | 75.4% |
| 0.30 | 83.3% | 70.8% |
| 0.32 | 86.0% | 66.5% |
| 0.34 | 88.5% | 63.3% |
| 0.36 | 91.0% | 58.9% |
| 0.38 | 93.0% | 55.5% |
| 0.40 | 94.0% | 52.8% |
| 0.42 | 95.4% | 50.2% |
| 0.44 | 96.4% | 48.3% |
| 0.46 | 97.0% | 46.2% |
| 0.48 | 97.4% | 43.5% |
| 0.50 | 97.8% | 41.4% |
| 0.52 | 98.4% | 39.1% |
| 0.54 | 98.8% | 36.9% |
| 0.56 | 99.2% | 34.6% |
| 0.58 | 99.3% | 32.3% |
| 0.60 | 99.3% | 29.9% |

## Confidence (rule 7)

Type: lowest threshold with at least 80.0% accuracy at or above it, never below the plan's starting 0.45: 0.45 (overall type accuracy 86.2%).

Queue and priority: the lowest pair where queue and priority are both right for at least 80.0% of the tickets that get a validated draft.

| queue | priority | validated drafts | queue and priority right among them |
|---|---|---|---|
| 0.45 | 0.45 | 16.4% | 65.2% |
| 0.45 | 0.55 | 12.6% | 69.7% |
| 0.55 | 0.45 | 10.8% | 71.3% |
| 0.45 | 0.65 | 8.4% | 78.1% |
| 0.55 | 0.55 | 8.0% | 77.0% |
| 0.65 | 0.45 | 6.7% | 76.0% |
| 0.45 | 0.75 | 4.9% | 87.4% |
| 0.75 | 0.45 | 4.4% | 73.0% |
| 0.55 | 0.65 | 5.5% | 84.8% |
| 0.65 | 0.55 | 4.9% | 80.5% |
| 0.55 | 0.75 | 3.5% | 91.4% |
| 0.65 | 0.65 | 3.6% | 87.8% |
| 0.75 | 0.55 | 3.2% | 78.4% |
| 0.65 | 0.75 | 2.4% | 94.6% |
| 0.75 | 0.65 | 2.1% | 87.8% |
| 0.75 | 0.75 | 1.5% | 91.2% |

**Chosen:** type 0.45, queue 0.45, priority 0.75, sparse evidence 0.28.

| outcome | plan defaults (0.45 each, evidence 0.30) | chosen |
|---|---|---|
| draft_ready | 15.7% | 4.9% |
| draft_failed | 3.7% | 1.1% |
| review | 55.8% | 69.2% |
| escalate | 24.8% | 24.8% |
| needs_clarification | 0.0% | 0.0% |

draft_ready counts validated drafts only. No hand labels were available, so safety was not checked by a person; label agreement is a proxy. The drafts only ask for information that a cited past answer also asked for, so the worst case is an unhelpful draft, not an invented commitment.
