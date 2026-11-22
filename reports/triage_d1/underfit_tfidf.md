# Underfitting check — variant A (tfidf), D1 split

Verdict rules (per head): **underfitting** = fit−val macro-F1 gap < 0.05 and a larger C than the chosen one improves calib macro-F1 by ≥ 0.01; **data-limited** = val macro-F1 rises ≥ 0.01 from 50% to 100% of fit; **overfitting** = gap > 0.15; otherwise ok. The capacity sweep is diagnostic only. Learning-curve subsamples reuse the features fitted on the full fit partition.

| head | C | F1 fit | F1 calib | F1 val | gap | log-loss fit / val | best larger-C gain | LC 25% / 50% / 100% | verdict |
|---|---|---|---|---|---|---|---|---|---|
| type | 4.0 | 0.994 | 0.865 | 0.873 | +0.121 | 0.124 / 0.318 | +0.0061 | 0.816 / 0.839 / 0.873 | data-limited |
| queue | 4.0 | 0.959 | 0.514 | 0.553 | +0.406 | 0.555 / 1.318 | +0.0433 | 0.390 / 0.450 / 0.553 | overfitting, data-limited |
| priority | 4.0 | 0.979 | 0.586 | 0.606 | +0.372 | 0.325 / 0.860 | +0.0325 | 0.472 / 0.529 / 0.606 | overfitting, data-limited |

Capacity sweep (calib macro-F1):

- type: C=1.0: 0.8541, C=4.0: 0.8655, C=16.0: 0.8716, C=64.0: 0.8704
- queue: C=1.0: 0.4667, C=4.0: 0.5136, C=16.0: 0.5570, C=64.0: 0.5546
- priority: C=1.0: 0.5589, C=4.0: 0.5857, C=16.0: 0.6115, C=64.0: 0.6182
