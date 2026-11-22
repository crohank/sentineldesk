# Scores: d2_tfidf_svm_tuned_extra

- predictions: `data/predictions/d2_tfidf_svm_tuned_extra.parquet`
- sha256: `70cb351438e38ca806259b581ef3706e7d3e112a60e31c5b5660da6f414fd756`
- rows: 5715
- scored at: 2026-11-21T20:51:50Z
- git commit: 3d812d1

Rates are shown as value (numerator/denominator).

## Summary

| metric | full |
|---|---|
| type macro-F1 | 0.874 |
| type accuracy | 0.867 (4955/5715) |
| queue macro-F1 | 0.605 |
| queue accuracy | 0.608 (3476/5715) |
| priority macro-F1 | 0.628 |
| priority accuracy | 0.641 (3664/5715) |
| Incident+high recall, flag on pred Incident and pred high | 0.655 (686/1047) |
| Incident+high precision, flag on pred Incident and pred high | 0.632 (686/1086) |
| Incident+high recall, flag on pred high alone | 0.733 (767/1047) |
| Incident+high precision, flag on pred high alone | 0.329 (767/2329) |
| type ECE, 10 bins | 0.011 |
| queue ECE, 10 bins | 0.029 |
| priority ECE, 10 bins | 0.030 |
| true Incident+high routed to human | 0.882 (923/1047) |
| true Incident+high routed to on-call | 0.778 (815/1047) |
| overall escalation rate | 0.580 (3316/5715) |
| overall on-call rate | 0.265 (1516/5715) |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.874 | 0.867 (4955/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.605 | 0.608 (3476/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.628 | 0.641 (3664/5715) |

## Per-class: type

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Change | 567 | 0.977 (554/567) | 0.977 (554/567) | 0.977 |
| Incident | 2321 | 0.840 (1953/2326) | 0.841 (1953/2321) | 0.841 |
| Problem | 1168 | 0.687 (800/1165) | 0.685 (800/1168) | 0.686 |
| Request | 1659 | 0.995 (1648/1657) | 0.993 (1648/1659) | 0.994 |

Confusion matrix: [confusion_type.png](confusion_type.png), [confusion_type.csv](confusion_type.csv)

## Per-class: queue

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Billing and Payments | 568 | 0.836 (450/538) | 0.792 (450/568) | 0.814 |
| Customer Service | 840 | 0.505 (488/967) | 0.581 (488/840) | 0.540 |
| General Inquiry | 77 | 0.660 (33/50) | 0.429 (33/77) | 0.520 |
| Human Resources | 115 | 0.775 (69/89) | 0.600 (69/115) | 0.676 |
| IT Support | 654 | 0.506 (361/714) | 0.552 (361/654) | 0.528 |
| Product Support | 1020 | 0.550 (553/1006) | 0.542 (553/1020) | 0.546 |
| Returns and Exchanges | 315 | 0.580 (152/262) | 0.483 (152/315) | 0.527 |
| Sales and Pre-Sales | 193 | 0.596 (84/141) | 0.435 (84/193) | 0.503 |
| Service Outages and Maintenance | 225 | 0.749 (164/219) | 0.729 (164/225) | 0.739 |
| Technical Support | 1708 | 0.649 (1122/1729) | 0.657 (1122/1708) | 0.653 |

Confusion matrix: [confusion_queue.png](confusion_queue.png), [confusion_queue.csv](confusion_queue.csv)

## Per-class: priority

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| high | 2294 | 0.673 (1567/2329) | 0.683 (1567/2294) | 0.678 |
| low | 1124 | 0.579 (620/1071) | 0.552 (620/1124) | 0.565 |
| medium | 2297 | 0.638 (1477/2315) | 0.643 (1477/2297) | 0.641 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 1086 | 0.655 (686/1047) | 0.632 (686/1086) |
| flag on pred high alone | 2329 | 0.733 (767/1047) | 0.329 (767/2329) |

- true events routed to a human (escalate_oncall or human_review): 0.882 (923/1047)
- true events routed to on-call: 0.778 (815/1047)
- overall escalation rate (all tickets to a human): 0.580 (3316/5715)
- overall on-call rate: 0.265 (1516/5715)
- decision_status counts: {'auto_draft': 2399, 'human_review': 1800, 'escalate_oncall': 1516}

## Calibration (top-label confidence)

`conf_<head>` is the confidence in the predicted label. 10 equal-width bins, each (lo, hi]. ECE = sum over bins of (bin n / N) * |bin accuracy - bin mean confidence|. Gap = accuracy minus mean confidence, so negative means overconfident.

| head | n | mean confidence | accuracy | ECE | plot |
|---|---|---|---|---|---|
| type | 5715 | 0.873 | 0.867 (4955/5715) | 0.011 | [reliability_type.png](reliability_type.png) |
| queue | 5715 | 0.585 | 0.608 (3476/5715) | 0.029 | [reliability_queue.png](reliability_queue.png) |
| priority | 5715 | 0.629 | 0.641 (3664/5715) | 0.030 | [reliability_priority.png](reliability_priority.png) |

### Bins: type

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 10 | 0.369 | 0.700 (7/10) | +0.331 |
| (0.4, 0.5] | 130 | 0.473 | 0.515 (67/130) | +0.042 |
| (0.5, 0.6] | 372 | 0.554 | 0.548 (204/372) | -0.005 |
| (0.6, 0.7] | 404 | 0.653 | 0.614 (248/404) | -0.039 |
| (0.7, 0.8] | 522 | 0.754 | 0.713 (372/522) | -0.042 |
| (0.8, 0.9] | 796 | 0.854 | 0.862 (686/796) | +0.007 |
| (0.9, 1.0] | 3481 | 0.971 | 0.968 (3371/3481) | -0.003 |

### Bins: queue

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 31 | 0.187 | 0.226 (7/31) | +0.039 |
| (0.2, 0.3] | 536 | 0.262 | 0.244 (131/536) | -0.017 |
| (0.3, 0.4] | 915 | 0.352 | 0.346 (317/915) | -0.006 |
| (0.4, 0.5] | 958 | 0.448 | 0.502 (481/958) | +0.054 |
| (0.5, 0.6] | 732 | 0.547 | 0.598 (438/732) | +0.051 |
| (0.6, 0.7] | 661 | 0.649 | 0.707 (467/661) | +0.058 |
| (0.7, 0.8] | 585 | 0.749 | 0.779 (456/585) | +0.031 |
| (0.8, 0.9] | 597 | 0.850 | 0.849 (507/597) | -0.001 |
| (0.9, 1.0] | 700 | 0.958 | 0.960 (672/700) | +0.002 |

### Bins: priority

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 246 | 0.378 | 0.346 (85/246) | -0.032 |
| (0.4, 0.5] | 1151 | 0.455 | 0.458 (527/1151) | +0.003 |
| (0.5, 0.6] | 1265 | 0.549 | 0.571 (722/1265) | +0.022 |
| (0.6, 0.7] | 1157 | 0.651 | 0.689 (797/1157) | +0.038 |
| (0.7, 0.8] | 950 | 0.748 | 0.796 (756/950) | +0.048 |
| (0.8, 0.9] | 708 | 0.846 | 0.806 (571/708) | -0.040 |
| (0.9, 1.0] | 238 | 0.931 | 0.866 (206/238) | -0.065 |

### ECE by language detected from the body

| language | n | type ECE | queue ECE | priority ECE |
|---|---|---|---|---|
| en | 3937 | 0.011 | 0.051 | 0.057 |
| de | 1770 | 0.021 | 0.037 | 0.064 |
| other | 8 | 0.139 | 0.297 | 0.363 |

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.892 | 0.671 | 0.684 | 0.702 (507/722) | 0.773 (558/722) |
| de | 1770 | 0.835 | 0.451 | 0.499 | 0.551 (178/323) | 0.641 (207/323) |
| other | 8 | 0.617 | 0.146 | 0.750 | 0.500 (1/2) | 1.000 (2/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.901 | 0.685 | 0.706 | 0.724 (430/594) | 0.798 (474/594) |
| de | 2431 | 0.840 | 0.485 | 0.520 | 0.565 (256/453) | 0.647 (293/453) |
| other | 0 | | | | | |

## Latency (ms)

- {'n': 5715, 'p50': 1.1242965879265632, 'p95': 1.1242965879265632, 'max': 1.1242965879265632}
