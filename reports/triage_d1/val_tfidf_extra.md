# Triage variant A +kb_extra (tfidf_extra) — D1 split, val partition

fit=29718 calib=2286 val=2286; head=lr, scaled=False

| head | C | converged (iters, warnings) | macro-F1 | acc | ECE raw | ECE cal | T |
|---|---|---|---|---|---|---|---|
| type | 4.0 | True (126, 0) | 0.864 | 0.853 | 0.015 | 0.005 | 0.91 |
| queue | 4.0 | True (176, 0) | 0.530 | 0.532 | 0.050 | 0.012 | 0.87 |
| priority | 4.0 | True (187, 0) | 0.606 | 0.617 | 0.018 | 0.020 | 1.12 |

## Outage (type==Incident & priority==high)

- gold in val: 389
- classifier recall 0.661, precision 0.591
- rule 6 (not met (best attainable recall)): P(Incident) ≥ 0.3, P(high) ≥ 0.3 → recall 0.843, precision 0.492, escalates 29.2%

Latency per ticket (predict incl. featurization): p50 23.9 ms, p95 35.8 ms
