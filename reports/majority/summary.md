# Summary: majority

- predictions: `data/predictions/majority.parquet`
- sha256: `568ec78e1763012713346071c34db9f74ad4d9dd9e39042d15b508792d1a696d`
- git commit: c98c534-dirty
- scored at: 2026-11-21T16:20:44Z

- full: [reports/majority/metrics.md](../majority/metrics.md)
- no near-dups: [reports/majority_no_near_dups/metrics.md](../majority_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/majority_no_twins/metrics.md](../majority_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.144 | 0.144 | 0.146 |
| type accuracy | 0.406 (2321/5715) | 0.406 (2318/5703) | 0.413 (2147/5195) |
| queue macro-F1 | 0.046 | 0.046 | 0.047 |
| queue accuracy | 0.299 (1708/5715) | 0.299 (1705/5703) | 0.303 (1576/5195) |
| priority macro-F1 | 0.191 | 0.191 | 0.192 |
| priority accuracy | 0.402 (2297/5715) | 0.402 (2293/5703) | 0.404 (2097/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.000 (0/1047) | 0.000 (0/1045) | 0.000 (0/967) |
| Incident+high precision, flag on pred Incident and pred high | n/a (0/0) | n/a (0/0) | n/a (0/0) |
| Incident+high recall, flag on pred high alone | 0.000 (0/1047) | 0.000 (0/1045) | 0.000 (0/967) |
| Incident+high precision, flag on pred high alone | n/a (0/0) | n/a (0/0) | n/a (0/0) |
