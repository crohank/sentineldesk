# Summary: tfidf_prio_german_A

- predictions: `data/predictions/tfidf_prio_german_A.parquet`
- sha256: `c303499b28efc2d60af06618c7f180d355d64ad78a6af7e7db1669e56715c80e`
- git commit: 1a23ec9
- scored at: 2026-11-21T16:38:14Z

- full: [reports/tfidf_prio_german_A/metrics.md](../tfidf_prio_german_A/metrics.md)
- no near-dups: [reports/tfidf_prio_german_A_no_near_dups/metrics.md](../tfidf_prio_german_A_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/tfidf_prio_german_A_no_twins/metrics.md](../tfidf_prio_german_A_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.884 | 0.884 | 0.876 |
| type accuracy | 0.876 (5004/5715) | 0.875 (4992/5703) | 0.866 (4498/5195) |
| queue macro-F1 | 0.587 | 0.587 | 0.555 |
| queue accuracy | 0.575 (3286/5715) | 0.575 (3277/5703) | 0.546 (2836/5195) |
| priority macro-F1 | 0.629 | 0.629 | 0.600 |
| priority accuracy | 0.643 (3674/5715) | 0.642 (3663/5703) | 0.617 (3203/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.648 (678/1047) | 0.647 (676/1045) | 0.630 (609/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.631 (678/1074) | 0.631 (676/1072) | 0.611 (609/997) |
| Incident+high recall, flag on pred high alone | 0.728 (762/1047) | 0.727 (760/1045) | 0.714 (690/967) |
| Incident+high precision, flag on pred high alone | 0.342 (762/2227) | 0.342 (760/2222) | 0.340 (690/2027) |
| type ECE, 10 bins | 0.026 | 0.025 | 0.020 |
| queue ECE, 10 bins | 0.081 | 0.080 | 0.062 |
| priority ECE, 10 bins | 0.021 | 0.020 | 0.028 |
