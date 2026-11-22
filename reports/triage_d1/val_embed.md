# Triage variant B (embed) — D1 split, val partition

fit=18285 calib=2286 val=2286

| head | C | converged (iters) | macro-F1 | acc | ECE raw | ECE cal | T |
|---|---|---|---|---|---|---|---|
| type | 10.0 | True (122) | 0.810 | 0.785 | 0.019 | 0.019 | 1.04 |
| queue | 10.0 | True (283) | 0.314 | 0.327 | 0.084 | 0.040 | 1.22 |
| priority | 10.0 | True (189) | 0.448 | 0.455 | 0.058 | 0.030 | 1.32 |

## Outage (type==Incident & priority==high)

- gold in val: 389
- classifier recall 0.555, precision 0.502
- rule 6 (not met (best attainable recall)): P(Incident) ≥ 0.3, P(high) ≥ 0.3 → recall 0.787, precision 0.412, escalates 32.5%

Latency per ticket (predict incl. featurization): p50 110.8 ms, p95 134.6 ms
