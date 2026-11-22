# Summary: d2_tfidf_svm_tuned_extra

- predictions: `data/predictions/d2_tfidf_svm_tuned_extra.parquet`
- sha256: `70cb351438e38ca806259b581ef3706e7d3e112a60e31c5b5660da6f414fd756`
- git commit: 3d812d1
- scored at: 2026-11-21T20:51:50Z

- full: [reports/d2_tfidf_svm_tuned_extra/metrics.md](../d2_tfidf_svm_tuned_extra/metrics.md)
- no near-dups: [reports/d2_tfidf_svm_tuned_extra_no_near_dups/metrics.md](../d2_tfidf_svm_tuned_extra_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/d2_tfidf_svm_tuned_extra_no_twins/metrics.md](../d2_tfidf_svm_tuned_extra_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.874 | 0.874 | 0.865 |
| type accuracy | 0.867 (4955/5715) | 0.867 (4943/5703) | 0.856 (4447/5195) |
| queue macro-F1 | 0.605 | 0.604 | 0.570 |
| queue accuracy | 0.608 (3476/5715) | 0.607 (3464/5703) | 0.578 (3001/5195) |
| priority macro-F1 | 0.628 | 0.627 | 0.599 |
| priority accuracy | 0.641 (3664/5715) | 0.641 (3654/5703) | 0.615 (3193/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.655 (686/1047) | 0.655 (684/1045) | 0.637 (616/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.632 (686/1086) | 0.631 (684/1084) | 0.611 (616/1009) |
| Incident+high recall, flag on pred high alone | 0.733 (767/1047) | 0.732 (765/1045) | 0.720 (696/967) |
| Incident+high precision, flag on pred high alone | 0.329 (767/2329) | 0.329 (765/2324) | 0.328 (696/2123) |
| type ECE, 10 bins | 0.011 | 0.011 | 0.015 |
| queue ECE, 10 bins | 0.029 | 0.029 | 0.026 |
| priority ECE, 10 bins | 0.030 | 0.030 | 0.021 |
| true Incident+high routed to human | 0.882 (923/1047) | 0.881 (921/1045) | 0.876 (847/967) |
| true Incident+high routed to on-call | 0.778 (815/1047) | 0.778 (813/1045) | 0.765 (740/967) |
| overall escalation rate | 0.580 (3316/5715) | 0.581 (3311/5703) | 0.609 (3162/5195) |
| overall on-call rate | 0.265 (1516/5715) | 0.265 (1514/5703) | 0.273 (1417/5195) |

Latency, full holdout: p50 1 ms, p95 1 ms, max 1 ms (n 5715).
