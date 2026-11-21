# Scores: tfidf_logreg_no_twins

- predictions: `data/predictions/tfidf_logreg.parquet`
- sha256: `bc39736a98fc72f526dd2810a85fc3b51fbefa2b45d35708a972a1957599bb74`
- rows: 5715
- scored at: 2026-11-21T15:59:14Z
- git commit: 5bd89c6
- excluded IDs: 520 from `reports/twin_holdout_ids.txt`

Rates are shown as value (numerator/denominator).

## Summary

| metric | full | excluding listed IDs |
|---|---|---|
| type macro-F1 | 0.884 | 0.876 |
| type accuracy | 0.876 (5004/5715) | 0.866 (4498/5195) |
| queue macro-F1 | 0.587 | 0.555 |
| queue accuracy | 0.575 (3286/5715) | 0.546 (2836/5195) |
| priority macro-F1 | 0.631 | 0.602 |
| priority accuracy | 0.642 (3670/5715) | 0.615 (3195/5195) |
| Incident+high recall, flag on pred Incident and pred high | 0.654 (685/1047) | 0.636 (615/967) |
| Incident+high precision, flag on pred Incident and pred high | 0.632 (685/1084) | 0.611 (615/1007) |
| Incident+high recall, flag on pred high alone | 0.735 (770/1047) | 0.722 (698/967) |
| Incident+high precision, flag on pred high alone | 0.342 (770/2253) | 0.340 (698/2054) |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.884 | 0.876 (5004/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.587 | 0.575 (3286/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.631 | 0.642 (3670/5715) |

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
| high | 2294 | 0.687 (1547/2253) | 0.674 (1547/2294) | 0.680 |
| low | 1124 | 0.548 (680/1242) | 0.605 (680/1124) | 0.575 |
| medium | 2297 | 0.650 (1443/2220) | 0.628 (1443/2297) | 0.639 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 1084 | 0.654 (685/1047) | 0.632 (685/1084) |
| flag on pred high alone | 2253 | 0.735 (770/1047) | 0.342 (770/2253) |

- routing: no decision_status column

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.897 | 0.643 | 0.682 | 0.676 (488/722) | 0.753 (544/722) |
| de | 1770 | 0.855 | 0.455 | 0.518 | 0.607 (196/323) | 0.697 (225/323) |
| other | 8 | 0.859 | 0.171 | 0.624 | 0.500 (1/2) | 0.500 (1/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.901 | 0.654 | 0.695 | 0.699 (415/594) | 0.771 (458/594) |
| de | 2431 | 0.863 | 0.489 | 0.545 | 0.596 (270/453) | 0.689 (312/453) |
| other | 0 | | | | | |

Per-class numbers for the excluded subset are in metrics.json under `excluded`.
