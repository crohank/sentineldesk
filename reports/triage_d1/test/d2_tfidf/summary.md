# Summary: d2_tfidf

- predictions: `data/predictions/d2_tfidf.parquet`
- sha256: `1de65586a2e19b895b78dece37d900e5062a12c2ccee0be38734f8c455edab73`
- git commit: 3d812d1
- scored at: 2026-11-21T20:49:17Z

- full: [reports/d2_tfidf/metrics.md](../d2_tfidf/metrics.md)
- no near-dups: [reports/d2_tfidf_no_near_dups/metrics.md](../d2_tfidf_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/d2_tfidf_no_twins/metrics.md](../d2_tfidf_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.872 | 0.872 | 0.864 |
| type accuracy | 0.862 (4926/5715) | 0.862 (4914/5703) | 0.852 (4428/5195) |
| queue macro-F1 | 0.558 | 0.558 | 0.529 |
| queue accuracy | 0.548 (3129/5715) | 0.548 (3124/5703) | 0.523 (2717/5195) |
| priority macro-F1 | 0.600 | 0.600 | 0.575 |
| priority accuracy | 0.612 (3496/5715) | 0.611 (3487/5703) | 0.588 (3057/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.625 (654/1047) | 0.624 (652/1045) | 0.612 (592/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.608 (654/1076) | 0.607 (652/1074) | 0.590 (592/1003) |
| Incident+high recall, flag on pred high alone | 0.717 (751/1047) | 0.717 (749/1045) | 0.708 (685/967) |
| Incident+high precision, flag on pred high alone | 0.339 (751/2217) | 0.339 (749/2211) | 0.339 (685/2020) |
| type ECE, 10 bins | 0.003 | 0.003 | 0.005 |
| queue ECE, 10 bins | 0.021 | 0.022 | 0.017 |
| priority ECE, 10 bins | 0.020 | 0.021 | 0.013 |
| true Incident+high routed to human | 0.904 (947/1047) | 0.904 (945/1045) | 0.900 (870/967) |
| true Incident+high routed to on-call | 0.793 (830/1047) | 0.792 (828/1045) | 0.781 (755/967) |
| overall escalation rate | 0.652 (3725/5715) | 0.652 (3719/5703) | 0.674 (3503/5195) |
| overall on-call rate | 0.285 (1628/5715) | 0.285 (1626/5703) | 0.293 (1522/5195) |

Latency, full holdout: p50 1 ms, p95 1 ms, max 1 ms (n 5715).
