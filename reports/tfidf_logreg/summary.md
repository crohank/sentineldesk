# Summary: tfidf_logreg

- predictions: `data/predictions/tfidf_logreg.parquet`
- sha256: `bc39736a98fc72f526dd2810a85fc3b51fbefa2b45d35708a972a1957599bb74`
- git commit: c98c534-dirty
- scored at: 2026-11-21T16:20:29Z

- full: [reports/tfidf_logreg/metrics.md](../tfidf_logreg/metrics.md)
- no near-dups: [reports/tfidf_logreg_no_near_dups/metrics.md](../tfidf_logreg_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/tfidf_logreg_no_twins/metrics.md](../tfidf_logreg_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.884 | 0.884 | 0.876 |
| type accuracy | 0.876 (5004/5715) | 0.875 (4992/5703) | 0.866 (4498/5195) |
| queue macro-F1 | 0.587 | 0.587 | 0.555 |
| queue accuracy | 0.575 (3286/5715) | 0.575 (3277/5703) | 0.546 (2836/5195) |
| priority macro-F1 | 0.631 | 0.631 | 0.602 |
| priority accuracy | 0.642 (3670/5715) | 0.641 (3658/5703) | 0.615 (3195/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.654 (685/1047) | 0.654 (683/1045) | 0.636 (615/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.632 (685/1084) | 0.631 (683/1082) | 0.611 (615/1007) |
| Incident+high recall, flag on pred high alone | 0.735 (770/1047) | 0.735 (768/1045) | 0.722 (698/967) |
| Incident+high precision, flag on pred high alone | 0.342 (770/2253) | 0.342 (768/2248) | 0.340 (698/2054) |
| type ECE, 10 bins | 0.026 | 0.025 | 0.020 |
| queue ECE, 10 bins | 0.081 | 0.080 | 0.062 |
| priority ECE, 10 bins | 0.018 | 0.018 | 0.021 |
