# Summary: d2_tfidf_extra

- predictions: `data/predictions/d2_tfidf_extra.parquet`
- sha256: `87b9a38f0ad39f880b89076a37561eb1d4813bc7976865da15430e917d51b36e`
- git commit: 3d812d1
- scored at: 2026-11-21T20:49:58Z

- full: [reports/d2_tfidf_extra/metrics.md](../d2_tfidf_extra/metrics.md)
- no near-dups: [reports/d2_tfidf_extra_no_near_dups/metrics.md](../d2_tfidf_extra_no_near_dups/metrics.md), excluding 12 IDs in `reports/near_dup_holdout_ids.txt`
- no twins: [reports/d2_tfidf_extra_no_twins/metrics.md](../d2_tfidf_extra_no_twins/metrics.md), excluding 520 IDs in `reports/twin_holdout_ids.txt`

Rates are value (numerator/denominator). ECE uses 10 equal-width bins on the confidence of the predicted label.

| metric | full | no near-dups | no twins |
|---|---|---|---|
| tickets scored | 5715 | 5703 | 5195 |
| type macro-F1 | 0.866 | 0.866 | 0.859 |
| type accuracy | 0.856 (4891/5715) | 0.856 (4879/5703) | 0.846 (4393/5195) |
| queue macro-F1 | 0.545 | 0.545 | 0.516 |
| queue accuracy | 0.542 (3100/5715) | 0.543 (3094/5703) | 0.518 (2690/5195) |
| priority macro-F1 | 0.594 | 0.594 | 0.568 |
| priority accuracy | 0.605 (3460/5715) | 0.605 (3450/5703) | 0.581 (3020/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.633 (663/1047) | 0.633 (661/1045) | 0.618 (598/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.617 (663/1075) | 0.616 (661/1073) | 0.600 (598/997) |
| Incident+high recall, flag on pred high alone | 0.722 (756/1047) | 0.722 (754/1045) | 0.709 (686/967) |
| Incident+high precision, flag on pred high alone | 0.337 (756/2243) | 0.337 (754/2238) | 0.336 (686/2043) |
| type ECE, 10 bins | 0.002 | 0.002 | 0.006 |
| queue ECE, 10 bins | 0.021 | 0.022 | 0.018 |
| priority ECE, 10 bins | 0.016 | 0.016 | 0.025 |
| true Incident+high routed to human | 0.918 (961/1047) | 0.918 (959/1045) | 0.913 (883/967) |
| true Incident+high routed to on-call | 0.796 (833/1047) | 0.795 (831/1045) | 0.785 (759/967) |
| overall escalation rate | 0.667 (3812/5715) | 0.667 (3806/5703) | 0.687 (3571/5195) |
| overall on-call rate | 0.281 (1608/5715) | 0.282 (1606/5703) | 0.288 (1496/5195) |

Latency, full holdout: p50 1 ms, p95 1 ms, max 1 ms (n 5715).
