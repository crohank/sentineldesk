# Scores: tfidf_logreg_plus20k

- predictions: `data/predictions/tfidf_logreg_plus20k.parquet`
- sha256: `a7ed87bfadcd6ed22fe173942db30825da2b8f1e2bbfb049e4ac241030425d49`
- rows: 5715
- scored at: 2026-11-21T16:33:29Z
- git commit: c394e41-dirty

Rates are shown as value (numerator/denominator).

## Summary

| metric | full |
|---|---|
| type macro-F1 | 0.879 |
| type accuracy | 0.868 (4962/5715) |
| queue macro-F1 | 0.569 |
| queue accuracy | 0.571 (3261/5715) |
| priority macro-F1 | 0.624 |
| priority accuracy | 0.635 (3628/5715) |
| Incident+high recall, flag on pred Incident and pred high | 0.651 (682/1047) |
| Incident+high precision, flag on pred Incident and pred high | 0.631 (682/1080) |
| Incident+high recall, flag on pred high alone | 0.741 (776/1047) |
| Incident+high precision, flag on pred high alone | 0.347 (776/2236) |
| type ECE, 10 bins | 0.022 |
| queue ECE, 10 bins | 0.080 |
| priority ECE, 10 bins | 0.017 |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.879 | 0.868 (4962/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.569 | 0.571 (3261/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.624 | 0.635 (3628/5715) |

## Per-class: type

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Change | 567 | 0.972 (562/578) | 0.991 (562/567) | 0.982 |
| Incident | 2321 | 0.863 (1885/2183) | 0.812 (1885/2321) | 0.837 |
| Problem | 1168 | 0.668 (869/1301) | 0.744 (869/1168) | 0.704 |
| Request | 1659 | 0.996 (1646/1653) | 0.992 (1646/1659) | 0.994 |

Confusion matrix: [confusion_type.png](confusion_type.png), [confusion_type.csv](confusion_type.csv)

## Per-class: queue

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Billing and Payments | 568 | 0.798 (442/554) | 0.778 (442/568) | 0.788 |
| Customer Service | 840 | 0.469 (454/969) | 0.540 (454/840) | 0.502 |
| General Inquiry | 77 | 0.464 (32/69) | 0.416 (32/77) | 0.438 |
| Human Resources | 115 | 0.703 (78/111) | 0.678 (78/115) | 0.690 |
| IT Support | 654 | 0.477 (356/747) | 0.544 (356/654) | 0.508 |
| Product Support | 1020 | 0.524 (516/985) | 0.506 (516/1020) | 0.515 |
| Returns and Exchanges | 315 | 0.485 (149/307) | 0.473 (149/315) | 0.479 |
| Sales and Pre-Sales | 193 | 0.434 (99/228) | 0.513 (99/193) | 0.470 |
| Service Outages and Maintenance | 225 | 0.629 (176/280) | 0.782 (176/225) | 0.697 |
| Technical Support | 1708 | 0.655 (959/1465) | 0.561 (959/1708) | 0.604 |

Confusion matrix: [confusion_queue.png](confusion_queue.png), [confusion_queue.csv](confusion_queue.csv)

## Per-class: priority

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| high | 2294 | 0.686 (1535/2236) | 0.669 (1535/2294) | 0.678 |
| low | 1124 | 0.531 (674/1269) | 0.600 (674/1124) | 0.563 |
| medium | 2297 | 0.642 (1419/2210) | 0.618 (1419/2297) | 0.630 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 1080 | 0.651 (682/1047) | 0.631 (682/1080) |
| flag on pred high alone | 2236 | 0.741 (776/1047) | 0.347 (776/2236) |

- routing: no decision_status column

## Calibration (top-label confidence)

`conf_<head>` is the confidence in the predicted label. 10 equal-width bins, each (lo, hi]. ECE = sum over bins of (bin n / N) * |bin accuracy - bin mean confidence|. Gap = accuracy minus mean confidence, so negative means overconfident.

| head | n | mean confidence | accuracy | ECE | plot |
|---|---|---|---|---|---|
| type | 5715 | 0.847 | 0.868 (4962/5715) | 0.022 | [reliability_type.png](reliability_type.png) |
| queue | 5715 | 0.491 | 0.571 (3261/5715) | 0.080 | [reliability_queue.png](reliability_queue.png) |
| priority | 5715 | 0.640 | 0.635 (3628/5715) | 0.017 | [reliability_priority.png](reliability_priority.png) |

### Bins: type

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 1 | 0.288 | 0.000 (0/1) | -0.288 |
| (0.3, 0.4] | 8 | 0.378 | 0.625 (5/8) | +0.247 |
| (0.4, 0.5] | 57 | 0.475 | 0.544 (31/57) | +0.069 |
| (0.5, 0.6] | 571 | 0.549 | 0.562 (321/571) | +0.013 |
| (0.6, 0.7] | 627 | 0.649 | 0.665 (417/627) | +0.016 |
| (0.7, 0.8] | 664 | 0.750 | 0.806 (535/664) | +0.056 |
| (0.8, 0.9] | 858 | 0.853 | 0.883 (758/858) | +0.030 |
| (0.9, 1.0] | 2929 | 0.976 | 0.988 (2895/2929) | +0.012 |

### Bins: queue

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 32 | 0.183 | 0.219 (7/32) | +0.035 |
| (0.2, 0.3] | 870 | 0.260 | 0.297 (258/870) | +0.037 |
| (0.3, 0.4] | 1475 | 0.350 | 0.431 (635/1475) | +0.080 |
| (0.4, 0.5] | 1115 | 0.449 | 0.543 (605/1115) | +0.094 |
| (0.5, 0.6] | 819 | 0.546 | 0.651 (533/819) | +0.105 |
| (0.6, 0.7] | 493 | 0.646 | 0.775 (382/493) | +0.129 |
| (0.7, 0.8] | 325 | 0.746 | 0.849 (276/325) | +0.104 |
| (0.8, 0.9] | 178 | 0.846 | 0.910 (162/178) | +0.064 |
| (0.9, 1.0] | 408 | 0.974 | 0.988 (403/408) | +0.014 |

### Bins: priority

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| (0.0, 0.1] | 0 | | | |
| (0.1, 0.2] | 0 | | | |
| (0.2, 0.3] | 0 | | | |
| (0.3, 0.4] | 212 | 0.377 | 0.387 (82/212) | +0.010 |
| (0.4, 0.5] | 1018 | 0.456 | 0.446 (454/1018) | -0.010 |
| (0.5, 0.6] | 1325 | 0.549 | 0.537 (711/1325) | -0.012 |
| (0.6, 0.7] | 1119 | 0.650 | 0.676 (757/1119) | +0.026 |
| (0.7, 0.8] | 997 | 0.749 | 0.752 (750/997) | +0.003 |
| (0.8, 0.9] | 735 | 0.847 | 0.823 (605/735) | -0.023 |
| (0.9, 1.0] | 309 | 0.937 | 0.871 (269/309) | -0.067 |

### ECE by language detected from the body

| language | n | type ECE | queue ECE | priority ECE |
|---|---|---|---|---|
| en | 3937 | 0.029 | 0.101 | 0.037 |
| de | 1770 | 0.017 | 0.034 | 0.083 |
| other | 8 | 0.213 | 0.264 | 0.191 |

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.894 | 0.624 | 0.672 | 0.668 (482/722) | 0.755 (545/722) |
| de | 1770 | 0.845 | 0.440 | 0.513 | 0.616 (199/323) | 0.712 (230/323) |
| other | 8 | 0.859 | 0.183 | 0.624 | 0.500 (1/2) | 0.500 (1/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.898 | 0.637 | 0.690 | 0.689 (409/594) | 0.774 (460/594) |
| de | 2431 | 0.854 | 0.469 | 0.533 | 0.603 (273/453) | 0.698 (316/453) |
| other | 0 | | | | | |
