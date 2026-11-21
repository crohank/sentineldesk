# Scores: tfidf_prio_german_E_no_near_dups

- predictions: `data/predictions/tfidf_prio_german_E.parquet`
- sha256: `80723db0fa4ed3b351194de5b2bccb335336cead5800af399b978715ca805422`
- rows: 5715
- scored at: 2026-11-21T17:29:20Z
- git commit: 200bcdb-dirty
- excluded IDs: 12 from `reports/near_dup_holdout_ids.txt`

Rates are shown as value (numerator/denominator).

## Summary

| metric | full | excluding listed IDs |
|---|---|---|
| type macro-F1 | 0.884 | 0.884 |
| type accuracy | 0.876 (5004/5715) | 0.875 (4992/5703) |
| queue macro-F1 | 0.587 | 0.587 |
| queue accuracy | 0.575 (3286/5715) | 0.575 (3277/5703) |
| priority macro-F1 | 0.633 | 0.632 |
| priority accuracy | 0.644 (3680/5715) | 0.643 (3668/5703) |
| Incident+high recall, flag on pred Incident and pred high | 0.652 (683/1047) | 0.652 (681/1045) |
| Incident+high precision, flag on pred Incident and pred high | 0.632 (683/1080) | 0.632 (681/1078) |
| Incident+high recall, flag on pred high alone | 0.736 (771/1047) | 0.736 (769/1045) |
| Incident+high precision, flag on pred high alone | 0.341 (771/2263) | 0.341 (769/2258) |
| type ECE, 10 bins | 0.026 | 0.025 |
| queue ECE, 10 bins | 0.081 | 0.080 |
| priority ECE, 10 bins | 0.021 | 0.021 |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.884 | 0.876 (5004/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.587 | 0.575 (3286/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.633 | 0.644 (3680/5715) |

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
| high | 2294 | 0.688 (1558/2263) | 0.679 (1558/2294) | 0.684 |
| low | 1124 | 0.550 (674/1225) | 0.600 (674/1124) | 0.574 |
| medium | 2297 | 0.650 (1448/2227) | 0.630 (1448/2297) | 0.640 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 1080 | 0.652 (683/1047) | 0.632 (683/1080) |
| flag on pred high alone | 2263 | 0.736 (771/1047) | 0.341 (771/2263) |

- routing: no decision_status column

## Calibration (top-label confidence)

`conf_<head>` is the confidence in the predicted label. 10 equal-width bins, each (lo, hi]. ECE = sum over bins of (bin n / N) * |bin accuracy - bin mean confidence|. Gap = accuracy minus mean confidence, so negative means overconfident.

| head | n | mean confidence | accuracy | ECE | plot |
|---|---|---|---|---|---|
| type | 5715 | 0.850 | 0.876 (5004/5715) | 0.026 | [reliability_type.png](reliability_type.png) |
| queue | 5715 | 0.495 | 0.575 (3286/5715) | 0.081 | [reliability_queue.png](reliability_queue.png) |
| priority | 5715 | 0.646 | 0.644 (3680/5715) | 0.021 | [reliability_priority.png](reliability_priority.png) |

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
| (0.3, 0.4] | 200 | 0.378 | 0.440 (88/200) | +0.062 |
| (0.4, 0.5] | 966 | 0.456 | 0.435 (420/966) | -0.022 |
| (0.5, 0.6] | 1275 | 0.551 | 0.541 (690/1275) | -0.010 |
| (0.6, 0.7] | 1175 | 0.651 | 0.669 (786/1175) | +0.018 |
| (0.7, 0.8] | 986 | 0.748 | 0.768 (757/986) | +0.020 |
| (0.8, 0.9] | 775 | 0.846 | 0.830 (643/775) | -0.017 |
| (0.9, 1.0] | 338 | 0.938 | 0.876 (296/338) | -0.062 |

### ECE by language detected from the body

| language | n | type ECE | queue ECE | priority ECE |
|---|---|---|---|---|
| en | 3937 | 0.031 | 0.102 | 0.041 |
| de | 1770 | 0.014 | 0.035 | 0.086 |
| other | 8 | 0.225 | 0.361 | 0.177 |

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.897 | 0.643 | 0.683 | 0.673 (486/722) | 0.753 (544/722) |
| de | 1770 | 0.855 | 0.455 | 0.519 | 0.607 (196/323) | 0.700 (226/323) |
| other | 8 | 0.859 | 0.171 | 0.624 | 0.500 (1/2) | 0.500 (1/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.901 | 0.654 | 0.697 | 0.699 (415/594) | 0.774 (460/594) |
| de | 2431 | 0.863 | 0.489 | 0.544 | 0.592 (268/453) | 0.687 (311/453) |
| other | 0 | | | | | |

Per-class numbers for the excluded subset are in metrics.json under `excluded`.
