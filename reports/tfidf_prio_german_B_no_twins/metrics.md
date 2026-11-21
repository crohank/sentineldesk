# Scores: tfidf_prio_german_B_no_twins

- predictions: `data/predictions/tfidf_prio_german_B.parquet`
- sha256: `8d0035584af704ef9f83dc3855dd949cbd7409c3dd86392498fa9ef2ff0240d1`
- rows: 5715
- scored at: 2026-11-21T17:05:56Z
- git commit: 567d741
- excluded IDs: 520 from `reports/twin_holdout_ids.txt`

Rates are shown as value (numerator/denominator).

## Summary

| metric | full | excluding listed IDs |
|---|---|---|
| type macro-F1 | 0.884 | 0.876 |
| type accuracy | 0.876 (5004/5715) | 0.866 (4498/5195) |
| queue macro-F1 | 0.587 | 0.555 |
| queue accuracy | 0.575 (3286/5715) | 0.546 (2836/5195) |
| priority macro-F1 | 0.630 | 0.601 |
| priority accuracy | 0.643 (3673/5715) | 0.616 (3200/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.657 (688/1047) | 0.639 (618/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.625 (688/1101) | 0.604 (618/1023) |
| Incident+high recall, flag on pred high alone | 0.742 (777/1047) | 0.729 (705/967) |
| Incident+high precision, flag on pred high alone | 0.337 (777/2305) | 0.335 (705/2103) |
| type ECE, 10 bins | 0.026 | 0.020 |
| queue ECE, 10 bins | 0.081 | 0.062 |
| priority ECE, 10 bins | 0.020 | 0.024 |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.884 | 0.876 (5004/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.587 | 0.575 (3286/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.630 | 0.643 (3673/5715) |

## Per-class: type

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Change | 567 | 0.972 (557/573) | 0.982 (557/567) | 0.977 |
| Incident | 2321 | 0.866 (1929/2228) | 0.831 (1929/2321) | 0.848 |
| Problem | 1168 | 0.692 (871/1259) | 0.746 (871/1168) | 0.718 |
| Request | 1659 | 0.995 (1647/1655) | 0.993 (1647/1659) | 0.994 |

Confusion matrix: [confusion_type.png](confusion_type.png), [confusion_type.csv](confusion_type.csv)

## Per-class: queue

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Billing and Payments | 568 | 0.806 (439/545) | 0.773 (439/568) | 0.789 |
| Customer Service | 840 | 0.469 (457/974) | 0.544 (457/840) | 0.504 |
| General Inquiry | 77 | 0.591 (39/66) | 0.506 (39/77) | 0.545 |
| Human Resources | 115 | 0.698 (74/106) | 0.643 (74/115) | 0.670 |
| IT Support | 654 | 0.472 (349/739) | 0.534 (349/654) | 0.501 |
| Product Support | 1020 | 0.515 (514/998) | 0.504 (514/1020) | 0.509 |
| Returns and Exchanges | 315 | 0.530 (160/302) | 0.508 (160/315) | 0.519 |
| Sales and Pre-Sales | 193 | 0.480 (98/204) | 0.508 (98/193) | 0.494 |
| Service Outages and Maintenance | 225 | 0.675 (179/265) | 0.796 (179/225) | 0.731 |
| Technical Support | 1708 | 0.644 (977/1516) | 0.572 (977/1708) | 0.606 |

Confusion matrix: [confusion_queue.png](confusion_queue.png), [confusion_queue.csv](confusion_queue.csv)

## Per-class: priority

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| high | 2294 | 0.682 (1573/2305) | 0.686 (1573/2294) | 0.684 |
| low | 1124 | 0.558 (647/1159) | 0.576 (647/1124) | 0.567 |
| medium | 2297 | 0.645 (1453/2251) | 0.633 (1453/2297) | 0.639 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 1101 | 0.657 (688/1047) | 0.625 (688/1101) |
| flag on pred high alone | 2305 | 0.742 (777/1047) | 0.337 (777/2305) |

- routing: no decision_status column

## Calibration (top-label confidence)

`conf_<head>` is the confidence in the predicted label. 10 equal-width bins, each (lo, hi]. ECE = sum over bins of (bin n / N) * |bin accuracy - bin mean confidence|. Gap = accuracy minus mean confidence, so negative means overconfident.

| head | n | mean confidence | accuracy | ECE | plot |
|---|---|---|---|---|---|
| type | 5715 | 0.850 | 0.876 (5004/5715) | 0.026 | [reliability_type.png](reliability_type.png) |
| queue | 5715 | 0.495 | 0.575 (3286/5715) | 0.081 | [reliability_queue.png](reliability_queue.png) |
| priority | 5715 | 0.647 | 0.643 (3673/5715) | 0.020 | [reliability_priority.png](reliability_priority.png) |

### Bins: type

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 7 | 0.380 | 0.571 (4/7) | +0.192 |
| (0.4, 0.5] | 61 | 0.476 | 0.459 (28/61) | -0.017 |
| (0.5, 0.6] | 520 | 0.553 | 0.604 (314/520) | +0.051 |
| (0.6, 0.7] | 618 | 0.650 | 0.696 (430/618) | +0.046 |
| (0.7, 0.8] | 698 | 0.750 | 0.792 (553/698) | +0.042 |
| (0.8, 0.9] | 842 | 0.853 | 0.892 (751/842) | +0.039 |
| (0.9, 1.0] | 2969 | 0.976 | 0.985 (2924/2969) | +0.009 |

### Bins: queue

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 30 | 0.185 | 0.167 (5/30) | -0.019 |
| (0.2, 0.3] | 818 | 0.261 | 0.292 (239/818) | +0.031 |
| (0.3, 0.4] | 1418 | 0.350 | 0.409 (580/1418) | +0.059 |
| (0.4, 0.5] | 1215 | 0.447 | 0.558 (678/1215) | +0.111 |
| (0.5, 0.6] | 765 | 0.547 | 0.671 (513/765) | +0.124 |
| (0.6, 0.7] | 534 | 0.647 | 0.768 (410/534) | +0.121 |
| (0.7, 0.8] | 364 | 0.746 | 0.843 (307/364) | +0.097 |
| (0.8, 0.9] | 171 | 0.846 | 0.906 (155/171) | +0.060 |
| (0.9, 1.0] | 400 | 0.972 | 0.998 (399/400) | +0.026 |

### Bins: priority

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 184 | 0.378 | 0.353 (65/184) | -0.025 |
| (0.4, 0.5] | 988 | 0.457 | 0.442 (437/988) | -0.015 |
| (0.5, 0.6] | 1245 | 0.551 | 0.545 (678/1245) | -0.006 |
| (0.6, 0.7] | 1174 | 0.649 | 0.675 (792/1174) | +0.025 |
| (0.7, 0.8] | 1027 | 0.748 | 0.762 (783/1027) | +0.014 |
| (0.8, 0.9] | 741 | 0.847 | 0.825 (611/741) | -0.023 |
| (0.9, 1.0] | 356 | 0.940 | 0.862 (307/356) | -0.077 |

### ECE by language detected from the body

| language | n | type ECE | queue ECE | priority ECE |
|---|---|---|---|---|
| en | 3937 | 0.031 | 0.102 | 0.041 |
| de | 1770 | 0.014 | 0.035 | 0.092 |
| other | 8 | 0.225 | 0.361 | 0.252 |

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.897 | 0.643 | 0.678 | 0.683 (493/722) | 0.763 (551/722) |
| de | 1770 | 0.855 | 0.455 | 0.522 | 0.598 (193/323) | 0.693 (224/323) |
| other | 8 | 0.859 | 0.171 | 0.750 | 1.000 (2/2) | 1.000 (2/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.901 | 0.654 | 0.692 | 0.705 (419/594) | 0.781 (464/594) |
| de | 2431 | 0.863 | 0.489 | 0.546 | 0.594 (269/453) | 0.691 (313/453) |
| other | 0 | | | | | |

Per-class numbers for the excluded subset are in metrics.json under `excluded`.
