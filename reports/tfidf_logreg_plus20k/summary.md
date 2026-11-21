# Summary: tfidf_logreg_plus20k

- predictions: `data/predictions/tfidf_logreg_plus20k.parquet`
- sha256: `a7ed87bfadcd6ed22fe173942db30825da2b8f1e2bbfb049e4ac241030425d49`
- git commit: c394e41-dirty
- scored at: 2026-11-21T16:33:29Z

- full: [reports/tfidf_logreg_plus20k/metrics.md](../tfidf_logreg_plus20k/metrics.md)
- no near-dups: [reports/tfidf_logreg_plus20k_no_near_dups/metrics.md](../tfidf_logreg_plus20k_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/tfidf_logreg_plus20k_no_twins/metrics.md](../tfidf_logreg_plus20k_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.879 | 0.879 | 0.870 |
| type accuracy | 0.868 (4962/5715) | 0.868 (4950/5703) | 0.857 (4453/5195) |
| queue macro-F1 | 0.569 | 0.569 | 0.538 |
| queue accuracy | 0.571 (3261/5715) | 0.570 (3252/5703) | 0.543 (2819/5195) |
| priority macro-F1 | 0.624 | 0.623 | 0.594 |
| priority accuracy | 0.635 (3628/5715) | 0.634 (3616/5703) | 0.607 (3155/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.651 (682/1047) | 0.651 (680/1045) | 0.632 (611/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.631 (682/1080) | 0.631 (680/1078) | 0.609 (611/1004) |
| Incident+high recall, flag on pred high alone | 0.741 (776/1047) | 0.741 (774/1045) | 0.727 (703/967) |
| Incident+high precision, flag on pred high alone | 0.347 (776/2236) | 0.347 (774/2231) | 0.344 (703/2041) |
| type ECE, 10 bins | 0.022 | 0.021 | 0.016 |
| queue ECE, 10 bins | 0.080 | 0.080 | 0.062 |
| priority ECE, 10 bins | 0.017 | 0.017 | 0.027 |
