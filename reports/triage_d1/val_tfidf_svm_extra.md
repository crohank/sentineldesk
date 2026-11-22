# Triage variant TF-IDF + LinearSVC +kb_extra (tfidf_svm_extra) — D1 split, val partition

fit=29718 calib=2286 val=2286; head=svm, scaled=False

| head | C | converged (iters, warnings) | macro-F1 | acc | ECE raw | ECE cal | T |
|---|---|---|---|---|---|---|---|
| type | 1.0 | True (78, 0) | 0.867 | 0.860 | 0.156 | 0.015 | 0.49 |
| queue | 1.0 | True (206, 0) | 0.585 | 0.587 | 0.324 | 0.023 | 0.34 |
| priority | 1.0 | True (85, 0) | 0.626 | 0.638 | 0.092 | 0.021 | 0.69 |

## Outage (type==Incident & priority==high)

- gold in val: 389
- classifier recall 0.697, precision 0.598
- rule 6 (not met (best attainable recall)): P(Incident) ≥ 0.3, P(high) ≥ 0.3 → recall 0.830, precision 0.512, escalates 27.6%

Latency per ticket (predict incl. featurization): p50 10.5 ms, p95 18.7 ms
