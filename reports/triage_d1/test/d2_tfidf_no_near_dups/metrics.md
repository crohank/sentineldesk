# Scores: d2_tfidf_no_near_dups

- predictions: `data/predictions/d2_tfidf.parquet`
- sha256: `1de65586a2e19b895b78dece37d900e5062a12c2ccee0be38734f8c455edab73`
- rows: 5715
- scored at: 2026-11-21T20:49:22Z
- git commit: 3d812d1
- excluded IDs: 12 from `reports/near_dup_holdout_ids.txt`

Rates are shown as value (numerator/denominator).

## Summary

| metric | full | excluding listed IDs |
|---|---|---|
| type macro-F1 | 0.872 | 0.872 |
| type accuracy | 0.862 (4926/5715) | 0.862 (4914/5703) |
| queue macro-F1 | 0.558 | 0.558 |
| queue accuracy | 0.548 (3129/5715) | 0.548 (3124/5703) |
| priority macro-F1 | 0.600 | 0.600 |
| priority accuracy | 0.612 (3496/5715) | 0.611 (3487/5703) |
| Incident+high recall, flag on pred Incident and pred high | 0.625 (654/1047) | 0.624 (652/1045) |
| Incident+high precision, flag on pred Incident and pred high | 0.608 (654/1076) | 0.607 (652/1074) |
| Incident+high recall, flag on pred high alone | 0.717 (751/1047) | 0.717 (749/1045) |
| Incident+high precision, flag on pred high alone | 0.339 (751/2217) | 0.339 (749/2211) |
| type ECE, 10 bins | 0.003 | 0.003 |
| queue ECE, 10 bins | 0.021 | 0.022 |
| priority ECE, 10 bins | 0.020 | 0.021 |
| true Incident+high routed to human | 0.904 (947/1047) | 0.904 (945/1045) |
| true Incident+high routed to on-call | 0.793 (830/1047) | 0.792 (828/1045) |
| overall escalation rate | 0.652 (3725/5715) | 0.652 (3719/5703) |
| overall on-call rate | 0.285 (1628/5715) | 0.285 (1626/5703) |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.872 | 0.862 (4926/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.558 | 0.548 (3129/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.600 | 0.612 (3496/5715) |

## Per-class: type

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Change | 567 | 0.974 (552/567) | 0.974 (552/567) | 0.974 |
| Incident | 2321 | 0.853 (1880/2205) | 0.810 (1880/2321) | 0.831 |
| Problem | 1168 | 0.658 (846/1285) | 0.724 (846/1168) | 0.690 |
| Request | 1659 | 0.994 (1648/1658) | 0.993 (1648/1659) | 0.994 |

Confusion matrix: [confusion_type.png](confusion_type.png), [confusion_type.csv](confusion_type.csv)

## Per-class: queue

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Billing and Payments | 568 | 0.809 (433/535) | 0.762 (433/568) | 0.785 |
| Customer Service | 840 | 0.439 (432/983) | 0.514 (432/840) | 0.474 |
| General Inquiry | 77 | 0.607 (37/61) | 0.481 (37/77) | 0.536 |
| Human Resources | 115 | 0.674 (62/92) | 0.539 (62/115) | 0.599 |
| IT Support | 654 | 0.439 (339/772) | 0.518 (339/654) | 0.475 |
| Product Support | 1020 | 0.474 (488/1029) | 0.478 (488/1020) | 0.476 |
| Returns and Exchanges | 315 | 0.503 (148/294) | 0.470 (148/315) | 0.486 |
| Sales and Pre-Sales | 193 | 0.431 (88/204) | 0.456 (88/193) | 0.443 |
| Service Outages and Maintenance | 225 | 0.677 (172/254) | 0.764 (172/225) | 0.718 |
| Technical Support | 1708 | 0.624 (930/1491) | 0.544 (930/1708) | 0.581 |

Confusion matrix: [confusion_queue.png](confusion_queue.png), [confusion_queue.csv](confusion_queue.csv)

## Per-class: priority

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| high | 2294 | 0.664 (1471/2217) | 0.641 (1471/2294) | 0.652 |
| low | 1124 | 0.514 (636/1237) | 0.566 (636/1124) | 0.539 |
| medium | 2297 | 0.614 (1389/2261) | 0.605 (1389/2297) | 0.609 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 1076 | 0.625 (654/1047) | 0.608 (654/1076) |
| flag on pred high alone | 2217 | 0.717 (751/1047) | 0.339 (751/2217) |

- true events routed to a human (escalate_oncall or human_review): 0.904 (947/1047)
- true events routed to on-call: 0.793 (830/1047)
- overall escalation rate (all tickets to a human): 0.652 (3725/5715)
- overall on-call rate: 0.285 (1628/5715)
- decision_status counts: {'human_review': 2097, 'auto_draft': 1990, 'escalate_oncall': 1628}

## Calibration (top-label confidence)

`conf_<head>` is the confidence in the predicted label. 10 equal-width bins, each (lo, hi]. ECE = sum over bins of (bin n / N) * |bin accuracy - bin mean confidence|. Gap = accuracy minus mean confidence, so negative means overconfident.

| head | n | mean confidence | accuracy | ECE | plot |
|---|---|---|---|---|---|
| type | 5715 | 0.862 | 0.862 (4926/5715) | 0.003 | [reliability_type.png](reliability_type.png) |
| queue | 5715 | 0.531 | 0.548 (3129/5715) | 0.021 | [reliability_queue.png](reliability_queue.png) |
| priority | 5715 | 0.600 | 0.612 (3496/5715) | 0.020 | [reliability_priority.png](reliability_priority.png) |

### Bins: type

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 5 | 0.361 | 0.200 (1/5) | -0.161 |
| (0.4, 0.5] | 36 | 0.469 | 0.444 (16/36) | -0.024 |
| (0.5, 0.6] | 520 | 0.551 | 0.562 (292/520) | +0.010 |
| (0.6, 0.7] | 549 | 0.650 | 0.652 (358/549) | +0.002 |
| (0.7, 0.8] | 621 | 0.752 | 0.758 (471/621) | +0.007 |
| (0.8, 0.9] | 820 | 0.854 | 0.850 (697/820) | -0.004 |
| (0.9, 1.0] | 3164 | 0.978 | 0.977 (3091/3164) | -0.001 |

### Bins: queue

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 24 | 0.186 | 0.125 (3/24) | -0.061 |
| (0.2, 0.3] | 532 | 0.264 | 0.246 (131/532) | -0.018 |
| (0.3, 0.4] | 1242 | 0.352 | 0.355 (441/1242) | +0.003 |
| (0.4, 0.5] | 1211 | 0.448 | 0.489 (592/1211) | +0.041 |
| (0.5, 0.6] | 901 | 0.548 | 0.559 (504/901) | +0.012 |
| (0.6, 0.7] | 610 | 0.647 | 0.697 (425/610) | +0.050 |
| (0.7, 0.8] | 463 | 0.747 | 0.765 (354/463) | +0.018 |
| (0.8, 0.9] | 274 | 0.847 | 0.832 (228/274) | -0.015 |
| (0.9, 1.0] | 458 | 0.973 | 0.985 (451/458) | +0.011 |

### Bins: priority

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 288 | 0.378 | 0.351 (101/288) | -0.027 |
| (0.4, 0.5] | 1322 | 0.455 | 0.453 (599/1322) | -0.002 |
| (0.5, 0.6] | 1484 | 0.549 | 0.557 (826/1484) | +0.008 |
| (0.6, 0.7] | 1182 | 0.647 | 0.694 (820/1182) | +0.047 |
| (0.7, 0.8] | 866 | 0.746 | 0.774 (670/866) | +0.028 |
| (0.8, 0.9] | 439 | 0.844 | 0.825 (362/439) | -0.019 |
| (0.9, 1.0] | 134 | 0.932 | 0.881 (118/134) | -0.051 |

### ECE by language detected from the body

| language | n | type ECE | queue ECE | priority ECE |
|---|---|---|---|---|
| en | 3937 | 0.006 | 0.032 | 0.049 |
| de | 1770 | 0.011 | 0.026 | 0.062 |
| other | 8 | 0.283 | 0.342 | 0.213 |

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.884 | 0.606 | 0.642 | 0.637 (460/722) | 0.731 (528/722) |
| de | 1770 | 0.843 | 0.441 | 0.505 | 0.598 (193/323) | 0.687 (222/323) |
| other | 8 | 0.859 | 0.183 | 0.490 | 0.500 (1/2) | 0.500 (1/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.889 | 0.617 | 0.656 | 0.660 (392/594) | 0.749 (445/594) |
| de | 2431 | 0.850 | 0.470 | 0.523 | 0.578 (262/453) | 0.675 (306/453) |
| other | 0 | | | | | |

## Latency (ms)

- {'n': 5715, 'p50': 1.1376059842519666, 'p95': 1.1376059842519666, 'max': 1.1376059842519666}

Per-class numbers for the excluded subset are in metrics.json under `excluded`.
