# Summary: tfidf_prio_german_D

- predictions: `data/predictions/tfidf_prio_german_D.parquet`
- sha256: `73075430f2f2709eb5ae5420719e2d4698d584430c699951dde583608e21ff46`
- git commit: 200bcdb-dirty
- scored at: 2026-11-21T17:29:05Z

- full: [reports/tfidf_prio_german_D/metrics.md](../tfidf_prio_german_D/metrics.md)
- no near-dups: [reports/tfidf_prio_german_D_no_near_dups/metrics.md](../tfidf_prio_german_D_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/tfidf_prio_german_D_no_twins/metrics.md](../tfidf_prio_german_D_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.884 | 0.884 | 0.876 |
| type accuracy | 0.876 (5004/5715) | 0.875 (4992/5703) | 0.866 (4498/5195) |
| queue macro-F1 | 0.587 | 0.587 | 0.555 |
| queue accuracy | 0.575 (3286/5715) | 0.575 (3277/5703) | 0.546 (2836/5195) |
| priority macro-F1 | 0.631 | 0.630 | 0.602 |
| priority accuracy | 0.643 (3673/5715) | 0.642 (3661/5703) | 0.616 (3200/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.652 (683/1047) | 0.652 (681/1045) | 0.634 (613/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.634 (683/1078) | 0.633 (681/1076) | 0.613 (613/1000) |
| Incident+high recall, flag on pred high alone | 0.736 (771/1047) | 0.736 (769/1045) | 0.723 (699/967) |
| Incident+high precision, flag on pred high alone | 0.343 (771/2247) | 0.343 (769/2242) | 0.342 (699/2045) |
| type ECE, 10 bins | 0.026 | 0.025 | 0.020 |
| queue ECE, 10 bins | 0.081 | 0.080 | 0.062 |
| priority ECE, 10 bins | 0.020 | 0.020 | 0.025 |
