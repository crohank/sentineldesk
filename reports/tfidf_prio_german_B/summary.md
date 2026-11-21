# Summary: tfidf_prio_german_B

- predictions: `data/predictions/tfidf_prio_german_B.parquet`
- sha256: `8d0035584af704ef9f83dc3855dd949cbd7409c3dd86392498fa9ef2ff0240d1`
- git commit: 567d741
- scored at: 2026-11-21T17:05:54Z

- full: [reports/tfidf_prio_german_B/metrics.md](../tfidf_prio_german_B/metrics.md)
- no near-dups: [reports/tfidf_prio_german_B_no_near_dups/metrics.md](../tfidf_prio_german_B_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/tfidf_prio_german_B_no_twins/metrics.md](../tfidf_prio_german_B_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.884 | 0.884 | 0.876 |
| type accuracy | 0.876 (5004/5715) | 0.875 (4992/5703) | 0.866 (4498/5195) |
| queue macro-F1 | 0.587 | 0.587 | 0.555 |
| queue accuracy | 0.575 (3286/5715) | 0.575 (3277/5703) | 0.546 (2836/5195) |
| priority macro-F1 | 0.630 | 0.629 | 0.601 |
| priority accuracy | 0.643 (3673/5715) | 0.642 (3661/5703) | 0.616 (3200/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.657 (688/1047) | 0.656 (686/1045) | 0.639 (618/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.625 (688/1101) | 0.624 (686/1099) | 0.604 (618/1023) |
| Incident+high recall, flag on pred high alone | 0.742 (777/1047) | 0.742 (775/1045) | 0.729 (705/967) |
| Incident+high precision, flag on pred high alone | 0.337 (777/2305) | 0.337 (775/2300) | 0.335 (705/2103) |
| type ECE, 10 bins | 0.026 | 0.025 | 0.020 |
| queue ECE, 10 bins | 0.081 | 0.080 | 0.062 |
| priority ECE, 10 bins | 0.020 | 0.020 | 0.024 |
