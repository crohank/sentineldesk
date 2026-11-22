# TF-IDF + LinearSVC hyperparameter search (D1 split, fit + kb_extra)

fit = 29718 (D1 fit 18285 + kb_extra 11433; 56 calib/val twins dropped), calib = 2286, val = 2286. TF-IDF features unchanged (173,656 columns). penalty=l2, dual=True. Selection by calib macro-F1 per head; everything in the comparison is on val.

ConvergenceWarnings remaining after raising max_iter (cap 64,000): none.

## 1. Full grid — calib macro-F1

### type

| class_weight | loss | C=0.1 | C=0.3 | C=1.0 | C=3.0 | C=10.0 |
|---|---|---|---|---|---|---|
| None | squared_hinge | 0.8330 | 0.8572 | 0.8760 | 0.8787 | 0.8797 |
| None | hinge | 0.7225 | 0.8146 | 0.8630 | 0.8790 | 0.8793 |
| balanced | squared_hinge | 0.8503 | 0.8615 | 0.8736 | **0.8801** | 0.8793 |
| balanced | hinge | 0.7981 | 0.8426 | 0.8670 | 0.8756 | 0.8778 |

### queue

| class_weight | loss | C=0.1 | C=0.3 | C=1.0 | C=3.0 | C=10.0 |
|---|---|---|---|---|---|---|
| None | squared_hinge | 0.4089 | 0.4882 | 0.5579 | 0.5655 | 0.5722 |
| None | hinge | 0.4589 | 0.4805 | 0.5229 | 0.5569 | 0.5698 |
| balanced | squared_hinge | 0.4592 | 0.5054 | 0.5609 | **0.5728** | 0.5721 |
| balanced | hinge | 0.3725 | 0.4120 | 0.4827 | 0.5532 | 0.5654 |

### priority

| class_weight | loss | C=0.1 | C=0.3 | C=1.0 | C=3.0 | C=10.0 | C=30.0 |
|---|---|---|---|---|---|---|---|
| None | squared_hinge | 0.5131 | 0.5754 | 0.6133 | 0.6259 | 0.6268 | 0.6282 |
| None | hinge | 0.4488 | 0.4884 | 0.5776 | 0.6195 | 0.6280 | — |
| balanced | squared_hinge | 0.5451 | 0.5891 | 0.6151 | 0.6254 | 0.6251 | **0.6282** |
| balanced | hinge | 0.5210 | 0.5437 | 0.5820 | 0.6236 | 0.6263 | — |

## 2. Selected settings

| head | C | class_weight | loss | calib macro-F1 | max_iter | T |
|---|---|---|---|---|---|---|
| type | 3.0 | balanced | squared_hinge | 0.8801 | 2000 | 0.559 |
| queue | 3.0 | balanced | squared_hinge | 0.5728 | 2000 | 0.427 |
| priority | 30.0 | balanced | squared_hinge | 0.6282 | 2000 | 1.079 |

- priority: C=10.0 was at the grid edge; added C=30.0 → selected C=30.0 (still at the new edge; not extended further). Note: 2/4 combinations at C=30.0 finished before the search was stopped (hinge-loss fits at C=30 not run).

## 3. Comparison on val (n = 2286)

| metric | A · TF-IDF + LR | SVM-tuned · TF-IDF + LinearSVC +kb_extra |
|---|---|---|
| macro-F1 type | 0.873 | 0.866 |
| macro-F1 queue | 0.553 | 0.609 |
| macro-F1 priority | 0.606 | 0.625 |
| ECE type | 0.012 | 0.020 |
| ECE queue | 0.019 | 0.029 |
| ECE priority | 0.024 | 0.019 |
| Incident∧high recall (classifier) | 0.653 | 0.697 |
| Incident∧high precision (classifier) | 0.577 | 0.602 |
| rule 6 best recall | 0.833 (@0.3/0.3) | 0.848 (@0.3/0.3) |
| rule 6 escalation rate | 29.3% | 27.3% |
| macro-F1 type · de | 0.846 | 0.826 |
| macro-F1 type · en | 0.893 | 0.893 |
| macro-F1 queue · de | 0.454 | 0.468 |
| macro-F1 queue · en | 0.609 | 0.696 |
| macro-F1 priority · de | 0.549 | 0.558 |
| macro-F1 priority · en | 0.647 | 0.674 |
| latency p95 per ticket | 18.0 ms | 6.7 ms |

Language rows use the dataset's `language` column (noisy). Val counts: {'en': 1330, 'de': 956}.

## 4. Decision

Rule (fixed): type or queue macro-F1 gain ≥ 0.02, no head drop > 0.01, p95 < 150 ms.

Δ vs A: type -0.008, queue +0.056, priority +0.019; p95 6.7 ms → **qualifies**.
