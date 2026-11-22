# Underfitting check — variant B (embed), D1 split

Verdict rules (per head): **underfitting** = fit−val macro-F1 gap < 0.05 and a larger C than the chosen one improves calib macro-F1 by ≥ 0.01; **data-limited** = val macro-F1 rises ≥ 0.01 from 50% to 100% of fit; **overfitting** = gap > 0.15; otherwise ok. The capacity sweep is diagnostic only. Learning-curve subsamples reuse the features fitted on the full fit partition.

| head | C | F1 fit | F1 calib | F1 val | gap | log-loss fit / val | best larger-C gain | LC 25% / 50% / 100% | verdict |
|---|---|---|---|---|---|---|---|---|---|
| type | 10.0 | 0.848 | 0.812 | 0.810 | +0.038 | 0.354 / 0.402 | -0.0026 | 0.790 / 0.803 / 0.810 | ok |
| queue | 10.0 | 0.405 | 0.318 | 0.314 | +0.091 | 1.577 / 1.776 | +0.0031 | 0.294 / 0.295 / 0.314 | data-limited |
| priority | 10.0 | 0.510 | 0.443 | 0.448 | +0.062 | 0.966 / 1.032 | +0.0137 | 0.435 / 0.441 / 0.448 | ok |

Capacity sweep (calib macro-F1):

- type: C=0.1: 0.8036, C=1.0: 0.8068, C=10.0: 0.8119, C=100.0: 0.8093, C=1000.0: 0.8076
- queue: C=0.1: 0.2739, C=1.0: 0.2970, C=10.0: 0.3180, C=100.0: 0.3211, C=1000.0: 0.3140
- priority: C=0.1: 0.4175, C=1.0: 0.4415, C=10.0: 0.4426, C=100.0: 0.4563, C=1000.0: 0.4546
