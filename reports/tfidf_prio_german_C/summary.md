# Summary: tfidf_prio_german_C

- predictions: `data/predictions/tfidf_prio_german_C.parquet`
- sha256: `4a89b7eb47148209dbbb8d729b9f7e346a72c9262f1bf1b347190353357d96eb`
- git commit: 1a23ec9
- scored at: 2026-11-21T16:38:43Z

- full: [reports/tfidf_prio_german_C/metrics.md](../tfidf_prio_german_C/metrics.md)
- no near-dups: [reports/tfidf_prio_german_C_no_near_dups/metrics.md](../tfidf_prio_german_C_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/tfidf_prio_german_C_no_twins/metrics.md](../tfidf_prio_german_C_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.884 | 0.884 | 0.876 |
| type accuracy | 0.876 (5004/5715) | 0.875 (4992/5703) | 0.866 (4498/5195) |
| queue macro-F1 | 0.587 | 0.587 | 0.555 |
| queue accuracy | 0.575 (3286/5715) | 0.575 (3277/5703) | 0.546 (2836/5195) |
| priority macro-F1 | 0.630 | 0.629 | 0.601 |
| priority accuracy | 0.643 (3672/5715) | 0.642 (3661/5703) | 0.616 (3200/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.651 (682/1047) | 0.651 (680/1045) | 0.634 (613/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.633 (682/1078) | 0.632 (680/1076) | 0.612 (613/1001) |
| Incident+high recall, flag on pred high alone | 0.733 (767/1047) | 0.732 (765/1045) | 0.719 (695/967) |
| Incident+high precision, flag on pred high alone | 0.343 (767/2238) | 0.343 (765/2233) | 0.341 (695/2037) |
| type ECE, 10 bins | 0.026 | 0.025 | 0.020 |
| queue ECE, 10 bins | 0.081 | 0.080 | 0.062 |
| priority ECE, 10 bins | 0.019 | 0.019 | 0.023 |
