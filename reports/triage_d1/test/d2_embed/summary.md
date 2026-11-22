# Summary: d2_embed

- predictions: `data/predictions/d2_embed.parquet`
- sha256: `fde7e67df7a469a63a5e86a125e8aee62668dbdd7aee0a6d58343fa2c2da4d36`
- git commit: 3d812d1
- scored at: 2026-11-21T20:50:37Z

- full: [reports/d2_embed/metrics.md](../d2_embed/metrics.md)
- no near-dups: [reports/d2_embed_no_near_dups/metrics.md](../d2_embed_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/d2_embed_no_twins/metrics.md](../d2_embed_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.826 | 0.825 | 0.824 |
| type accuracy | 0.804 (4593/5715) | 0.803 (4581/5703) | 0.800 (4154/5195) |
| queue macro-F1 | 0.325 | 0.326 | 0.321 |
| queue accuracy | 0.331 (1889/5715) | 0.331 (1886/5703) | 0.328 (1703/5195) |
| priority macro-F1 | 0.458 | 0.457 | 0.453 |
| priority accuracy | 0.467 (2668/5715) | 0.467 (2661/5703) | 0.464 (2409/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.516 (540/1047) | 0.515 (538/1045) | 0.511 (494/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.520 (540/1039) | 0.519 (538/1036) | 0.518 (494/954) |
| Incident+high recall, flag on pred high alone | 0.649 (680/1047) | 0.649 (678/1045) | 0.646 (625/967) |
| Incident+high precision, flag on pred high alone | 0.321 (680/2118) | 0.321 (678/2113) | 0.322 (625/1939) |
| type ECE, 10 bins | 0.007 | 0.007 | 0.009 |
| queue ECE, 10 bins | 0.038 | 0.038 | 0.039 |
| priority ECE, 10 bins | 0.011 | 0.010 | 0.011 |
| true Incident+high routed to human | 0.971 (1017/1047) | 0.971 (1015/1045) | 0.970 (938/967) |
| true Incident+high routed to on-call | 0.757 (793/1047) | 0.757 (791/1045) | 0.750 (725/967) |
| overall escalation rate | 0.928 (5303/5715) | 0.928 (5292/5703) | 0.931 (4839/5195) |
| overall on-call rate | 0.324 (1851/5715) | 0.324 (1847/5703) | 0.328 (1705/5195) |

Latency, full holdout: p50 26 ms, p95 26 ms, max 26 ms (n 5715).
