# Summary: d2_tfidf_svm_extra

- predictions: `data/predictions/d2_tfidf_svm_extra.parquet`
- sha256: `9001faca9a29956e47211e57ff67978b79cc149ed1888648498b5b52d4e20e04`
- git commit: 3d812d1
- scored at: 2026-11-21T20:51:13Z

- full: [reports/d2_tfidf_svm_extra/metrics.md](../d2_tfidf_svm_extra/metrics.md)
- no near-dups: [reports/d2_tfidf_svm_extra_no_near_dups/metrics.md](../d2_tfidf_svm_extra_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/d2_tfidf_svm_extra_no_twins/metrics.md](../d2_tfidf_svm_extra_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.871 | 0.871 | 0.863 |
| type accuracy | 0.864 (4936/5715) | 0.863 (4924/5703) | 0.854 (4435/5195) |
| queue macro-F1 | 0.591 | 0.591 | 0.559 |
| queue accuracy | 0.596 (3408/5715) | 0.596 (3399/5703) | 0.569 (2955/5195) |
| priority macro-F1 | 0.623 | 0.623 | 0.597 |
| priority accuracy | 0.635 (3630/5715) | 0.635 (3620/5703) | 0.611 (3176/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.670 (702/1047) | 0.670 (700/1045) | 0.655 (633/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.623 (702/1127) | 0.622 (700/1125) | 0.606 (633/1045) |
| Incident+high recall, flag on pred high alone | 0.749 (784/1047) | 0.748 (782/1045) | 0.737 (713/967) |
| Incident+high precision, flag on pred high alone | 0.338 (784/2319) | 0.338 (782/2314) | 0.337 (713/2114) |
| type ECE, 10 bins | 0.010 | 0.010 | 0.015 |
| queue ECE, 10 bins | 0.022 | 0.022 | 0.017 |
| priority ECE, 10 bins | 0.022 | 0.021 | 0.014 |
| true Incident+high routed to human | 0.897 (939/1047) | 0.897 (937/1045) | 0.892 (863/967) |
| true Incident+high routed to on-call | 0.781 (818/1047) | 0.781 (816/1045) | 0.770 (745/967) |
| overall escalation rate | 0.606 (3462/5715) | 0.606 (3455/5703) | 0.631 (3278/5195) |
| overall on-call rate | 0.272 (1553/5715) | 0.272 (1551/5703) | 0.279 (1449/5195) |

Latency, full holdout: p50 1 ms, p95 1 ms, max 1 ms (n 5715).
