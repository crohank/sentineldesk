# Triage variant A (tfidf) — D1 split, val partition

fit=18285 calib=2286 val=2286

| head | C | converged (iters) | macro-F1 | acc | ECE raw | ECE cal | T |
|---|---|---|---|---|---|---|---|
| type | 4.0 | True (77) | 0.873 | 0.863 | 0.023 | 0.012 | 0.88 |
| queue | 4.0 | True (160) | 0.553 | 0.541 | 0.060 | 0.019 | 0.86 |
| priority | 4.0 | True (105) | 0.606 | 0.616 | 0.016 | 0.024 | 1.17 |

## Outage (type==Incident & priority==high)

- gold in val: 389
- classifier recall 0.653, precision 0.577
- rule 6 (not met (best attainable recall)): P(Incident) ≥ 0.3, P(high) ≥ 0.3 → recall 0.833, precision 0.484, escalates 29.3%

Latency per ticket (predict incl. featurization): p50 26.6 ms, p95 38.1 ms
