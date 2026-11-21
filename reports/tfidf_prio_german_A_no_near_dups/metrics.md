# Scores: tfidf_prio_german_A_no_near_dups

- predictions: `data/predictions/tfidf_prio_german_A.parquet`
- sha256: `c303499b28efc2d60af06618c7f180d355d64ad78a6af7e7db1669e56715c80e`
- rows: 5715
- scored at: 2026-11-21T16:38:15Z
- git commit: 1a23ec9
- excluded IDs: 12 from `reports/near_dup_holdout_ids.txt`

Rates are shown as value (numerator/denominator).

## Summary

| metric | full | excluding listed IDs |
|---|---|---|
| type macro-F1 | 0.884 | 0.884 |
| type accuracy | 0.876 (5004/5715) | 0.875 (4992/5703) |
| queue macro-F1 | 0.587 | 0.587 |
| queue accuracy | 0.575 (3286/5715) | 0.575 (3277/5703) |
| priority macro-F1 | 0.629 | 0.629 |
| priority accuracy | 0.643 (3674/5715) | 0.642 (3663/5703) |
| Incident+high recall, flag on pred Incident and pred high | 0.648 (678/1047) | 0.647 (676/1045) |
| Incident+high precision, flag on pred Incident and pred high | 0.631 (678/1074) | 0.631 (676/1072) |
| Incident+high recall, flag on pred high alone | 0.728 (762/1047) | 0.727 (760/1045) |
| Incident+high precision, flag on pred high alone | 0.342 (762/2227) | 0.342 (760/2222) |
| type ECE, 10 bins | 0.026 | 0.025 |
| queue ECE, 10 bins | 0.081 | 0.080 |
| priority ECE, 10 bins | 0.021 | 0.020 |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.884 | 0.876 (5004/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.587 | 0.575 (3286/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.629 | 0.643 (3674/5715) |

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
| high | 2294 | 0.690 (1536/2227) | 0.670 (1536/2294) | 0.679 |
| low | 1124 | 0.573 (622/1085) | 0.553 (622/1124) | 0.563 |
| medium | 2297 | 0.631 (1516/2403) | 0.660 (1516/2297) | 0.645 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 1074 | 0.648 (678/1047) | 0.631 (678/1074) |
| flag on pred high alone | 2227 | 0.728 (762/1047) | 0.342 (762/2227) |

- routing: no decision_status column

## Calibration (top-label confidence)

`conf_<head>` is the confidence in the predicted label. 10 equal-width bins, each (lo, hi]. ECE = sum over bins of (bin n / N) * |bin accuracy - bin mean confidence|. Gap = accuracy minus mean confidence, so negative means overconfident.

| head | n | mean confidence | accuracy | ECE | plot |
|---|---|---|---|---|---|
| type | 5715 | 0.850 | 0.876 (5004/5715) | 0.026 | [reliability_type.png](reliability_type.png) |
| queue | 5715 | 0.495 | 0.575 (3286/5715) | 0.081 | [reliability_queue.png](reliability_queue.png) |
| priority | 5715 | 0.649 | 0.643 (3674/5715) | 0.021 | [reliability_priority.png](reliability_priority.png) |

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
| (0.3, 0.4] | 191 | 0.379 | 0.403 (77/191) | +0.024 |
| (0.4, 0.5] | 929 | 0.458 | 0.433 (402/929) | -0.025 |
| (0.5, 0.6] | 1285 | 0.549 | 0.557 (716/1285) | +0.009 |
| (0.6, 0.7] | 1173 | 0.651 | 0.659 (773/1173) | +0.008 |
| (0.7, 0.8] | 1029 | 0.748 | 0.765 (787/1029) | +0.017 |
| (0.8, 0.9] | 760 | 0.848 | 0.817 (621/760) | -0.031 |
| (0.9, 1.0] | 348 | 0.939 | 0.856 (298/348) | -0.083 |

### ECE by language detected from the body

| language | n | type ECE | queue ECE | priority ECE |
|---|---|---|---|---|
| en | 3937 | 0.031 | 0.102 | 0.043 |
| de | 1770 | 0.014 | 0.035 | 0.096 |
| other | 8 | 0.225 | 0.361 | 0.428 |

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.897 | 0.643 | 0.677 | 0.670 (484/722) | 0.745 (538/722) |
| de | 1770 | 0.855 | 0.455 | 0.522 | 0.594 (192/323) | 0.687 (222/323) |
| other | 8 | 0.859 | 0.171 | 0.750 | 1.000 (2/2) | 1.000 (2/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.901 | 0.654 | 0.692 | 0.695 (413/594) | 0.764 (454/594) |
| de | 2431 | 0.863 | 0.489 | 0.544 | 0.585 (265/453) | 0.680 (308/453) |
| other | 0 | | | | | |

Per-class numbers for the excluded subset are in metrics.json under `excluded`.
