# Triage — validation report (thr half of split811 val)

train=32213 cal=2013 thr=2014

| head | macro-F1 | acc | majority acc | ECE raw | ECE calibrated | T |
|---|---|---|---|---|---|---|
| type | 0.866 | 0.856 | 0.399 | 0.013 | 0.018 | 0.89 |
| queue | 0.552 | 0.556 | 0.285 | 0.071 | 0.028 | 0.84 |
| priority | 0.602 | 0.619 | 0.406 | 0.021 | 0.019 | 1.17 |

## Confidence threshold → coverage / accuracy on covered

| threshold | type cov | type acc | queue cov | queue acc | priority cov | priority acc |
|---|---|---|---|---|---|---|
| 0.30 | 1.00 | 0.856 | 0.91 | 0.577 | 1.00 | 0.619 |
| 0.35 | 1.00 | 0.856 | 0.81 | 0.612 | 1.00 | 0.619 |
| 0.40 | 1.00 | 0.856 | 0.70 | 0.653 | 0.96 | 0.626 |
| 0.45 | 1.00 | 0.857 | 0.59 | 0.692 | 0.86 | 0.648 |
| 0.50 | 0.99 | 0.860 | 0.50 | 0.733 | 0.72 | 0.679 |
| 0.55 | 0.95 | 0.869 | 0.40 | 0.770 | 0.57 | 0.712 |
| 0.60 | 0.90 | 0.884 | 0.32 | 0.810 | 0.44 | 0.758 |
| 0.65 | 0.86 | 0.895 | 0.27 | 0.857 | 0.35 | 0.778 |
| 0.70 | 0.81 | 0.912 | 0.21 | 0.882 | 0.25 | 0.798 |
| 0.75 | 0.76 | 0.924 | 0.17 | 0.921 | 0.18 | 0.828 |
| 0.80 | 0.70 | 0.934 | 0.14 | 0.938 | 0.11 | 0.862 |
| 0.85 | 0.62 | 0.953 | 0.10 | 0.966 | 0.06 | 0.900 |
| 0.90 | 0.54 | 0.971 | 0.08 | 0.988 | 0.03 | 0.897 |

## Outage (type==Incident & priority==high)

- gold on thr half: 347
- classifier joint prediction: recall 0.695 (241/347), precision 0.600
- probability rule, target recall 90% not met (best attainable recall): P(Incident) ≥ 0.3, P(high) ≥ 0.3
- combined gate (classifier OR rule): recall 0.821, precision 0.486, escalates 29.1% of tickets
