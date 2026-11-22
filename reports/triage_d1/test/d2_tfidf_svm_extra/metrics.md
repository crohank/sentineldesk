# Scores: d2_tfidf_svm_extra

- predictions: `data/predictions/d2_tfidf_svm_extra.parquet`
- sha256: `9001faca9a29956e47211e57ff67978b79cc149ed1888648498b5b52d4e20e04`
- rows: 5715
- scored at: 2026-11-21T20:51:13Z
- git commit: 3d812d1

Rates are shown as value (numerator/denominator).

## Summary

| metric | full |
|---|---|
| type macro-F1 | 0.871 |
| type accuracy | 0.864 (4936/5715) |
| queue macro-F1 | 0.591 |
| queue accuracy | 0.596 (3408/5715) |
| priority macro-F1 | 0.623 |
| priority accuracy | 0.635 (3630/5715) |
| Incident+high recall, flag on pred Incident and pred high | 0.670 (702/1047) |
| Incident+high precision, flag on pred Incident and pred high | 0.623 (702/1127) |
| Incident+high recall, flag on pred high alone | 0.749 (784/1047) |
| Incident+high precision, flag on pred high alone | 0.338 (784/2319) |
| type ECE, 10 bins | 0.010 |
| queue ECE, 10 bins | 0.022 |
| priority ECE, 10 bins | 0.022 |
| true Incident+high routed to human | 0.897 (939/1047) |
| true Incident+high routed to on-call | 0.781 (818/1047) |
| overall escalation rate | 0.606 (3462/5715) |
| overall on-call rate | 0.272 (1553/5715) |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.871 | 0.864 (4936/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.591 | 0.596 (3408/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.623 | 0.635 (3630/5715) |

## Per-class: type

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Change | 567 | 0.977 (557/570) | 0.982 (557/567) | 0.980 |
| Incident | 2321 | 0.834 (1948/2335) | 0.839 (1948/2321) | 0.837 |
| Problem | 1168 | 0.680 (784/1153) | 0.671 (784/1168) | 0.676 |
| Request | 1659 | 0.994 (1647/1657) | 0.993 (1647/1659) | 0.993 |

Confusion matrix: [confusion_type.png](confusion_type.png), [confusion_type.csv](confusion_type.csv)

## Per-class: queue

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Billing and Payments | 568 | 0.827 (449/543) | 0.790 (449/568) | 0.808 |
| Customer Service | 840 | 0.505 (480/951) | 0.571 (480/840) | 0.536 |
| General Inquiry | 77 | 0.642 (34/53) | 0.442 (34/77) | 0.523 |
| Human Resources | 115 | 0.679 (72/106) | 0.626 (72/115) | 0.652 |
| IT Support | 654 | 0.483 (360/746) | 0.550 (360/654) | 0.514 |
| Product Support | 1020 | 0.546 (523/957) | 0.513 (523/1020) | 0.529 |
| Returns and Exchanges | 315 | 0.531 (163/307) | 0.517 (163/315) | 0.524 |
| Sales and Pre-Sales | 193 | 0.497 (87/175) | 0.451 (87/193) | 0.473 |
| Service Outages and Maintenance | 225 | 0.672 (168/250) | 0.747 (168/225) | 0.707 |
| Technical Support | 1708 | 0.659 (1072/1627) | 0.628 (1072/1708) | 0.643 |

Confusion matrix: [confusion_queue.png](confusion_queue.png), [confusion_queue.csv](confusion_queue.csv)

## Per-class: priority

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| high | 2294 | 0.671 (1557/2319) | 0.679 (1557/2294) | 0.675 |
| low | 1124 | 0.566 (636/1123) | 0.566 (636/1124) | 0.566 |
| medium | 2297 | 0.632 (1437/2273) | 0.626 (1437/2297) | 0.629 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 1127 | 0.670 (702/1047) | 0.623 (702/1127) |
| flag on pred high alone | 2319 | 0.749 (784/1047) | 0.338 (784/2319) |

- true events routed to a human (escalate_oncall or human_review): 0.897 (939/1047)
- true events routed to on-call: 0.781 (818/1047)
- overall escalation rate (all tickets to a human): 0.606 (3462/5715)
- overall on-call rate: 0.272 (1553/5715)
- decision_status counts: {'auto_draft': 2253, 'human_review': 1909, 'escalate_oncall': 1553}

## Calibration (top-label confidence)

`conf_<head>` is the confidence in the predicted label. 10 equal-width bins, each (lo, hi]. ECE = sum over bins of (bin n / N) * |bin accuracy - bin mean confidence|. Gap = accuracy minus mean confidence, so negative means overconfident.

| head | n | mean confidence | accuracy | ECE | plot |
|---|---|---|---|---|---|
| type | 5715 | 0.871 | 0.864 (4936/5715) | 0.010 | [reliability_type.png](reliability_type.png) |
| queue | 5715 | 0.574 | 0.596 (3408/5715) | 0.022 | [reliability_queue.png](reliability_queue.png) |
| priority | 5715 | 0.624 | 0.635 (3630/5715) | 0.022 | [reliability_priority.png](reliability_priority.png) |

### Bins: type

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 11 | 0.373 | 0.636 (7/11) | +0.264 |
| (0.4, 0.5] | 114 | 0.471 | 0.509 (58/114) | +0.038 |
| (0.5, 0.6] | 403 | 0.552 | 0.541 (218/403) | -0.012 |
| (0.6, 0.7] | 448 | 0.654 | 0.607 (272/448) | -0.047 |
| (0.7, 0.8] | 505 | 0.754 | 0.735 (371/505) | -0.019 |
| (0.8, 0.9] | 807 | 0.855 | 0.845 (682/807) | -0.010 |
| (0.9, 1.0] | 3427 | 0.974 | 0.971 (3328/3427) | -0.002 |

### Bins: queue

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 24 | 0.187 | 0.208 (5/24) | +0.022 |
| (0.2, 0.3] | 571 | 0.261 | 0.294 (168/571) | +0.033 |
| (0.3, 0.4] | 1021 | 0.352 | 0.370 (378/1021) | +0.018 |
| (0.4, 0.5] | 985 | 0.450 | 0.499 (492/985) | +0.049 |
| (0.5, 0.6] | 748 | 0.547 | 0.567 (424/748) | +0.019 |
| (0.6, 0.7] | 612 | 0.645 | 0.668 (409/612) | +0.023 |
| (0.7, 0.8] | 537 | 0.752 | 0.756 (406/537) | +0.004 |
| (0.8, 0.9] | 525 | 0.852 | 0.869 (456/525) | +0.017 |
| (0.9, 1.0] | 692 | 0.966 | 0.968 (670/692) | +0.002 |

### Bins: priority

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 255 | 0.377 | 0.412 (105/255) | +0.035 |
| (0.4, 0.5] | 1196 | 0.455 | 0.466 (557/1196) | +0.010 |
| (0.5, 0.6] | 1269 | 0.549 | 0.562 (713/1269) | +0.013 |
| (0.6, 0.7] | 1194 | 0.650 | 0.676 (807/1194) | +0.026 |
| (0.7, 0.8] | 925 | 0.749 | 0.776 (718/925) | +0.027 |
| (0.8, 0.9] | 657 | 0.846 | 0.831 (546/657) | -0.015 |
| (0.9, 1.0] | 219 | 0.932 | 0.840 (184/219) | -0.092 |

### ECE by language detected from the body

| language | n | type ECE | queue ECE | priority ECE |
|---|---|---|---|---|
| en | 3937 | 0.007 | 0.046 | 0.055 |
| de | 1770 | 0.024 | 0.036 | 0.081 |
| other | 8 | 0.237 | 0.354 | 0.239 |

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.885 | 0.653 | 0.675 | 0.699 (505/722) | 0.776 (560/722) |
| de | 1770 | 0.841 | 0.448 | 0.505 | 0.607 (196/323) | 0.687 (222/323) |
| other | 8 | 0.617 | 0.148 | 0.524 | 0.500 (1/2) | 1.000 (2/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.892 | 0.667 | 0.693 | 0.715 (425/594) | 0.795 (472/594) |
| de | 2431 | 0.844 | 0.479 | 0.527 | 0.611 (277/453) | 0.689 (312/453) |
| other | 0 | | | | | |

## Latency (ms)

- {'n': 5715, 'p50': 1.0866050218722934, 'p95': 1.0866050218722934, 'max': 1.0866050218722934}
