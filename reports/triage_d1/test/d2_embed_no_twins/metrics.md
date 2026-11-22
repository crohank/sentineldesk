# Scores: d2_embed_no_twins

- predictions: `data/predictions/d2_embed.parquet`
- sha256: `fde7e67df7a469a63a5e86a125e8aee62668dbdd7aee0a6d58343fa2c2da4d36`
- rows: 5715
- scored at: 2026-11-21T20:50:44Z
- git commit: 3d812d1
- excluded IDs: 520 from `reports/twin_holdout_ids.txt`

Rates are shown as value (numerator/denominator).

## Summary

| metric | full | excluding listed IDs |
|---|---|---|
| type macro-F1 | 0.826 | 0.824 |
| type accuracy | 0.804 (4593/5715) | 0.800 (4154/5195) |
| queue macro-F1 | 0.325 | 0.321 |
| queue accuracy | 0.331 (1889/5715) | 0.328 (1703/5195) |
| priority macro-F1 | 0.458 | 0.453 |
| priority accuracy | 0.467 (2668/5715) | 0.464 (2409/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.516 (540/1047) | 0.511 (494/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.520 (540/1039) | 0.518 (494/954) |
| Incident+high recall, flag on pred high alone | 0.649 (680/1047) | 0.646 (625/967) |
| Incident+high precision, flag on pred high alone | 0.321 (680/2118) | 0.322 (625/1939) |
| type ECE, 10 bins | 0.007 | 0.009 |
| queue ECE, 10 bins | 0.038 | 0.039 |
| priority ECE, 10 bins | 0.011 | 0.011 |
| true Incident+high routed to human | 0.971 (1017/1047) | 0.970 (938/967) |
| true Incident+high routed to on-call | 0.757 (793/1047) | 0.750 (725/967) |
| overall escalation rate | 0.928 (5303/5715) | 0.931 (4839/5195) |
| overall on-call rate | 0.324 (1851/5715) | 0.328 (1705/5195) |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.826 | 0.804 (4593/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.325 | 0.331 (1889/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.458 | 0.467 (2668/5715) |

## Per-class: type

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Change | 567 | 0.949 (557/587) | 0.982 (557/567) | 0.965 |
| Incident | 2321 | 0.821 (1578/1922) | 0.680 (1578/2321) | 0.744 |
| Problem | 1168 | 0.527 (826/1567) | 0.707 (826/1168) | 0.604 |
| Request | 1659 | 0.996 (1632/1639) | 0.984 (1632/1659) | 0.990 |

Confusion matrix: [confusion_type.png](confusion_type.png), [confusion_type.csv](confusion_type.csv)

## Per-class: queue

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Billing and Payments | 568 | 0.711 (393/553) | 0.692 (393/568) | 0.701 |
| Customer Service | 840 | 0.342 (205/600) | 0.244 (205/840) | 0.285 |
| General Inquiry | 77 | 0.078 (38/486) | 0.494 (38/77) | 0.135 |
| Human Resources | 115 | 0.136 (63/463) | 0.548 (63/115) | 0.218 |
| IT Support | 654 | 0.265 (195/737) | 0.298 (195/654) | 0.280 |
| Product Support | 1020 | 0.324 (174/537) | 0.171 (174/1020) | 0.224 |
| Returns and Exchanges | 315 | 0.230 (124/539) | 0.394 (124/315) | 0.290 |
| Sales and Pre-Sales | 193 | 0.174 (91/523) | 0.472 (91/193) | 0.254 |
| Service Outages and Maintenance | 225 | 0.401 (173/431) | 0.769 (173/225) | 0.527 |
| Technical Support | 1708 | 0.512 (433/846) | 0.254 (433/1708) | 0.339 |

Confusion matrix: [confusion_queue.png](confusion_queue.png), [confusion_queue.csv](confusion_queue.csv)

## Per-class: priority

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| high | 2294 | 0.572 (1212/2118) | 0.528 (1212/2294) | 0.549 |
| low | 1124 | 0.310 (584/1884) | 0.520 (584/1124) | 0.388 |
| medium | 2297 | 0.509 (872/1713) | 0.380 (872/2297) | 0.435 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 1039 | 0.516 (540/1047) | 0.520 (540/1039) |
| flag on pred high alone | 2118 | 0.649 (680/1047) | 0.321 (680/2118) |

- true events routed to a human (escalate_oncall or human_review): 0.971 (1017/1047)
- true events routed to on-call: 0.757 (793/1047)
- overall escalation rate (all tickets to a human): 0.928 (5303/5715)
- overall on-call rate: 0.324 (1851/5715)
- decision_status counts: {'human_review': 3452, 'escalate_oncall': 1851, 'auto_draft': 412}

## Calibration (top-label confidence)

`conf_<head>` is the confidence in the predicted label. 10 equal-width bins, each (lo, hi]. ECE = sum over bins of (bin n / N) * |bin accuracy - bin mean confidence|. Gap = accuracy minus mean confidence, so negative means overconfident.

| head | n | mean confidence | accuracy | ECE | plot |
|---|---|---|---|---|---|
| type | 5715 | 0.802 | 0.804 (4593/5715) | 0.007 | [reliability_type.png](reliability_type.png) |
| queue | 5715 | 0.366 | 0.331 (1889/5715) | 0.038 | [reliability_queue.png](reliability_queue.png) |
| priority | 5715 | 0.469 | 0.467 (2668/5715) | 0.011 | [reliability_priority.png](reliability_priority.png) |

### Bins: type

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 5 | 0.362 | 0.600 (3/5) | +0.238 |
| (0.4, 0.5] | 38 | 0.473 | 0.421 (16/38) | -0.052 |
| (0.5, 0.6] | 971 | 0.550 | 0.553 (537/971) | +0.003 |
| (0.6, 0.7] | 954 | 0.650 | 0.653 (623/954) | +0.003 |
| (0.7, 0.8] | 875 | 0.748 | 0.734 (642/875) | -0.015 |
| (0.8, 0.9] | 589 | 0.847 | 0.864 (509/589) | +0.017 |
| (0.9, 1.0] | 2283 | 0.987 | 0.991 (2263/2283) | +0.005 |

### Bins: queue

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 489 | 0.182 | 0.190 (93/489) | +0.008 |
| (0.2, 0.3] | 2390 | 0.249 | 0.243 (581/2390) | -0.006 |
| (0.3, 0.4] | 1396 | 0.343 | 0.299 (417/1396) | -0.044 |
| (0.4, 0.5] | 550 | 0.443 | 0.355 (195/550) | -0.088 |
| (0.5, 0.6] | 243 | 0.541 | 0.333 (81/243) | -0.208 |
| (0.6, 0.7] | 134 | 0.644 | 0.478 (64/134) | -0.167 |
| (0.7, 0.8] | 91 | 0.742 | 0.604 (55/91) | -0.138 |
| (0.8, 0.9] | 75 | 0.856 | 0.840 (63/75) | -0.016 |
| (0.9, 1.0] | 347 | 0.972 | 0.980 (340/347) | +0.008 |

### Bins: priority

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 1384 | 0.377 | 0.379 (524/1384) | +0.002 |
| (0.4, 0.5] | 2713 | 0.445 | 0.446 (1211/2713) | +0.002 |
| (0.5, 0.6] | 1074 | 0.541 | 0.507 (545/1074) | -0.034 |
| (0.6, 0.7] | 361 | 0.640 | 0.670 (242/361) | +0.031 |
| (0.7, 0.8] | 148 | 0.744 | 0.777 (115/148) | +0.033 |
| (0.8, 0.9] | 34 | 0.828 | 0.882 (30/34) | +0.054 |
| (0.9, 1.0] | 1 | 0.902 | 1.000 (1/1) | +0.098 |

### ECE by language detected from the body

| language | n | type ECE | queue ECE | priority ECE |
|---|---|---|---|---|
| en | 3937 | 0.010 | 0.035 | 0.012 |
| de | 1770 | 0.015 | 0.048 | 0.028 |
| other | 8 | 0.448 | 0.336 | 0.132 |

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.829 | 0.334 | 0.462 | 0.521 (376/722) | 0.654 (472/722) |
| de | 1770 | 0.819 | 0.304 | 0.446 | 0.508 (164/323) | 0.644 (208/323) |
| other | 8 | 0.689 | 0.238 | 0.444 | 0.000 (0/2) | 0.000 (0/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.826 | 0.334 | 0.463 | 0.520 (309/594) | 0.660 (392/594) |
| de | 2431 | 0.826 | 0.313 | 0.450 | 0.510 (231/453) | 0.636 (288/453) |
| other | 0 | | | | | |

## Latency (ms)

- {'n': 5715, 'p50': 26.280331058617715, 'p95': 26.280331058617715, 'max': 26.280331058617715}

Per-class numbers for the excluded subset are in metrics.json under `excluded`.
