# Scores: d2_tfidf_extra_no_near_dups

- predictions: `data/predictions/d2_tfidf_extra.parquet`
- sha256: `87b9a38f0ad39f880b89076a37561eb1d4813bc7976865da15430e917d51b36e`
- rows: 5715
- scored at: 2026-11-21T20:50:02Z
- git commit: 3d812d1
- excluded IDs: 12 from `reports/near_dup_holdout_ids.txt`

Rates are shown as value (numerator/denominator).

## Summary

| metric | full | excluding listed IDs |
|---|---|---|
| type macro-F1 | 0.866 | 0.866 |
| type accuracy | 0.856 (4891/5715) | 0.856 (4879/5703) |
| queue macro-F1 | 0.545 | 0.545 |
| queue accuracy | 0.542 (3100/5715) | 0.543 (3094/5703) |
| priority macro-F1 | 0.594 | 0.594 |
| priority accuracy | 0.605 (3460/5715) | 0.605 (3450/5703) |
| Incident+high recall, flag on pred Incident and pred high | 0.633 (663/1047) | 0.633 (661/1045) |
| Incident+high precision, flag on pred Incident and pred high | 0.617 (663/1075) | 0.616 (661/1073) |
| Incident+high recall, flag on pred high alone | 0.722 (756/1047) | 0.722 (754/1045) |
| Incident+high precision, flag on pred high alone | 0.337 (756/2243) | 0.337 (754/2238) |
| type ECE, 10 bins | 0.002 | 0.002 |
| queue ECE, 10 bins | 0.021 | 0.022 |
| priority ECE, 10 bins | 0.016 | 0.016 |
| true Incident+high routed to human | 0.918 (961/1047) | 0.918 (959/1045) |
| true Incident+high routed to on-call | 0.796 (833/1047) | 0.795 (831/1045) |
| overall escalation rate | 0.667 (3812/5715) | 0.667 (3806/5703) |
| overall on-call rate | 0.281 (1608/5715) | 0.282 (1606/5703) |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.866 | 0.856 (4891/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.545 | 0.542 (3100/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.594 | 0.605 (3460/5715) |

## Per-class: type

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Change | 567 | 0.965 (559/579) | 0.986 (559/567) | 0.976 |
| Incident | 2321 | 0.845 (1864/2206) | 0.803 (1864/2321) | 0.824 |
| Problem | 1168 | 0.646 (824/1276) | 0.705 (824/1168) | 0.674 |
| Request | 1659 | 0.994 (1644/1654) | 0.991 (1644/1659) | 0.992 |

Confusion matrix: [confusion_type.png](confusion_type.png), [confusion_type.csv](confusion_type.csv)

## Per-class: queue

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Billing and Payments | 568 | 0.800 (437/546) | 0.769 (437/568) | 0.785 |
| Customer Service | 840 | 0.439 (439/1001) | 0.523 (439/840) | 0.477 |
| General Inquiry | 77 | 0.471 (33/70) | 0.429 (33/77) | 0.449 |
| Human Resources | 115 | 0.699 (65/93) | 0.565 (65/115) | 0.625 |
| IT Support | 654 | 0.435 (337/775) | 0.515 (337/654) | 0.472 |
| Product Support | 1020 | 0.481 (474/986) | 0.465 (474/1020) | 0.473 |
| Returns and Exchanges | 315 | 0.469 (143/305) | 0.454 (143/315) | 0.461 |
| Sales and Pre-Sales | 193 | 0.433 (93/215) | 0.482 (93/193) | 0.456 |
| Service Outages and Maintenance | 225 | 0.615 (169/275) | 0.751 (169/225) | 0.676 |
| Technical Support | 1708 | 0.628 (910/1449) | 0.533 (910/1708) | 0.576 |

Confusion matrix: [confusion_queue.png](confusion_queue.png), [confusion_queue.csv](confusion_queue.csv)

## Per-class: priority

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| high | 2294 | 0.658 (1477/2243) | 0.644 (1477/2294) | 0.651 |
| low | 1124 | 0.502 (639/1273) | 0.569 (639/1124) | 0.533 |
| medium | 2297 | 0.611 (1344/2199) | 0.585 (1344/2297) | 0.598 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 1075 | 0.633 (663/1047) | 0.617 (663/1075) |
| flag on pred high alone | 2243 | 0.722 (756/1047) | 0.337 (756/2243) |

- true events routed to a human (escalate_oncall or human_review): 0.918 (961/1047)
- true events routed to on-call: 0.796 (833/1047)
- overall escalation rate (all tickets to a human): 0.667 (3812/5715)
- overall on-call rate: 0.281 (1608/5715)
- decision_status counts: {'human_review': 2204, 'auto_draft': 1903, 'escalate_oncall': 1608}

## Calibration (top-label confidence)

`conf_<head>` is the confidence in the predicted label. 10 equal-width bins, each (lo, hi]. ECE = sum over bins of (bin n / N) * |bin accuracy - bin mean confidence|. Gap = accuracy minus mean confidence, so negative means overconfident.

| head | n | mean confidence | accuracy | ECE | plot |
|---|---|---|---|---|---|
| type | 5715 | 0.856 | 0.856 (4891/5715) | 0.002 | [reliability_type.png](reliability_type.png) |
| queue | 5715 | 0.524 | 0.542 (3100/5715) | 0.021 | [reliability_queue.png](reliability_queue.png) |
| priority | 5715 | 0.609 | 0.605 (3460/5715) | 0.016 | [reliability_priority.png](reliability_priority.png) |

### Bins: type

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 5 | 0.374 | 0.400 (2/5) | +0.026 |
| (0.4, 0.5] | 46 | 0.469 | 0.457 (21/46) | -0.012 |
| (0.5, 0.6] | 541 | 0.550 | 0.542 (293/541) | -0.008 |
| (0.6, 0.7] | 582 | 0.651 | 0.651 (379/582) | +0.000 |
| (0.7, 0.8] | 625 | 0.752 | 0.747 (467/625) | -0.004 |
| (0.8, 0.9] | 827 | 0.853 | 0.854 (706/827) | +0.001 |
| (0.9, 1.0] | 3089 | 0.977 | 0.979 (3023/3089) | +0.001 |

### Bins: queue

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 18 | 0.188 | 0.167 (3/18) | -0.021 |
| (0.2, 0.3] | 590 | 0.265 | 0.263 (155/590) | -0.002 |
| (0.3, 0.4] | 1340 | 0.352 | 0.384 (514/1340) | +0.032 |
| (0.4, 0.5] | 1177 | 0.448 | 0.483 (569/1177) | +0.036 |
| (0.5, 0.6] | 848 | 0.548 | 0.540 (458/848) | -0.008 |
| (0.6, 0.7] | 639 | 0.647 | 0.679 (434/639) | +0.032 |
| (0.7, 0.8] | 377 | 0.750 | 0.764 (288/377) | +0.014 |
| (0.8, 0.9] | 248 | 0.843 | 0.855 (212/248) | +0.012 |
| (0.9, 1.0] | 478 | 0.976 | 0.977 (467/478) | +0.001 |

### Bins: priority

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 267 | 0.377 | 0.363 (97/267) | -0.014 |
| (0.4, 0.5] | 1252 | 0.454 | 0.446 (558/1252) | -0.009 |
| (0.5, 0.6] | 1452 | 0.549 | 0.537 (780/1452) | -0.012 |
| (0.6, 0.7] | 1169 | 0.648 | 0.672 (785/1169) | +0.023 |
| (0.7, 0.8] | 894 | 0.747 | 0.755 (675/894) | +0.008 |
| (0.8, 0.9] | 515 | 0.845 | 0.817 (421/515) | -0.027 |
| (0.9, 1.0] | 166 | 0.934 | 0.867 (144/166) | -0.067 |

### ECE by language detected from the body

| language | n | type ECE | queue ECE | priority ECE |
|---|---|---|---|---|
| en | 3937 | 0.011 | 0.036 | 0.036 |
| de | 1770 | 0.020 | 0.023 | 0.080 |
| other | 8 | 0.199 | 0.512 | 0.341 |

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.879 | 0.595 | 0.639 | 0.657 (474/722) | 0.745 (538/722) |
| de | 1770 | 0.838 | 0.422 | 0.492 | 0.582 (188/323) | 0.669 (216/323) |
| other | 8 | 0.617 | 0.250 | 0.606 | 0.500 (1/2) | 1.000 (2/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.885 | 0.605 | 0.655 | 0.675 (401/594) | 0.769 (457/594) |
| de | 2431 | 0.842 | 0.453 | 0.511 | 0.578 (262/453) | 0.660 (299/453) |
| other | 0 | | | | | |

## Latency (ms)

- {'n': 5715, 'p50': 1.218487401574796, 'p95': 1.218487401574796, 'max': 1.218487401574796}

Per-class numbers for the excluded subset are in metrics.json under `excluded`.
