# Triage variant A [en only] (tfidf_en) — D1 split, val partition

fit=10399 calib=1314 val=1330; head=lr, scaled=False

| head | C | converged (iters, warnings) | macro-F1 | acc | ECE raw | ECE cal | T |
|---|---|---|---|---|---|---|---|
| type | 4.0 | True (55, 0) | 0.883 | 0.872 | 0.029 | 0.017 | 0.82 |
| queue | 4.0 | True (140, 0) | 0.607 | 0.586 | 0.093 | 0.031 | 0.80 |
| priority | 4.0 | True (117, 0) | 0.637 | 0.650 | 0.017 | 0.018 | 0.99 |

## Outage (type==Incident & priority==high)

- gold in val: 234
- classifier recall 0.679, precision 0.609
- rule 6 (not met (best attainable recall)): P(Incident) ≥ 0.3, P(high) ≥ 0.3 → recall 0.821, precision 0.520, escalates 27.7%

Latency per ticket (predict incl. featurization): p50 8.7 ms, p95 11.6 ms
