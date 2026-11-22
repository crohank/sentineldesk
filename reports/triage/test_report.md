# Triage — holdout report (split811 test)

n = 4027  ·  predictions sha256 `20e2da4c1cc48c22`

| head | macro-F1 | acc | majority macro-F1 | majority acc | ECE |
|---|---|---|---|---|---|
| type | 0.861 | 0.851 | 0.142 | 0.398 | 0.017 |
| queue | 0.544 | 0.559 | 0.044 | 0.282 | 0.023 |
| priority | 0.608 | 0.621 | 0.196 | 0.418 | 0.024 |

## High-priority Incident (type==Incident & priority==high)

- classifier recall **0.626** (432/690), precision 0.598 (432/723)
- classifier OR probability rule (P(Incident) ≥ 0.3, P(high) ≥ 0.3): recall 0.775 (535/690), precision 0.490, escalates 1091/4027 (27.1%)
