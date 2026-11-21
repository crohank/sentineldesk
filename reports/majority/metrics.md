# Scores: majority

- predictions: `data/predictions/majority.parquet`
- sha256: `568ec78e1763012713346071c34db9f74ad4d9dd9e39042d15b508792d1a696d`
- rows: 5715
- scored at: 2026-11-21T16:20:44Z
- git commit: c98c534-dirty

Rates are shown as value (numerator/denominator).

## Summary

| metric | full |
|---|---|
| type macro-F1 | 0.144 |
| type accuracy | 0.406 (2321/5715) |
| queue macro-F1 | 0.046 |
| queue accuracy | 0.299 (1708/5715) |
| priority macro-F1 | 0.191 |
| priority accuracy | 0.402 (2297/5715) |
| Incident+high recall, flag on pred Incident and pred high | 0.000 (0/1047) |
| Incident+high precision, flag on pred Incident and pred high | n/a (0/0) |
| Incident+high recall, flag on pred high alone | 0.000 (0/1047) |
| Incident+high precision, flag on pred high alone | n/a (0/0) |

## Against a training-majority predictor

| target | majority label | majority macro-F1 | majority accuracy | run macro-F1 | run accuracy |
|---|---|---|---|---|---|
| type | Incident | 0.144 | 0.406 (2321/5715) | 0.144 | 0.406 (2321/5715) |
| queue | Technical Support | 0.046 | 0.299 (1708/5715) | 0.046 | 0.299 (1708/5715) |
| priority | medium | 0.191 | 0.402 (2297/5715) | 0.191 | 0.402 (2297/5715) |

## Per-class: type

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Change | 567 | n/a (0/0) | 0.000 (0/567) | 0.000 |
| Incident | 2321 | 0.406 (2321/5715) | 1.000 (2321/2321) | 0.578 |
| Problem | 1168 | n/a (0/0) | 0.000 (0/1168) | 0.000 |
| Request | 1659 | n/a (0/0) | 0.000 (0/1659) | 0.000 |

Confusion matrix: [confusion_type.png](confusion_type.png), [confusion_type.csv](confusion_type.csv)

## Per-class: queue

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Billing and Payments | 568 | n/a (0/0) | 0.000 (0/568) | 0.000 |
| Customer Service | 840 | n/a (0/0) | 0.000 (0/840) | 0.000 |
| General Inquiry | 77 | n/a (0/0) | 0.000 (0/77) | 0.000 |
| Human Resources | 115 | n/a (0/0) | 0.000 (0/115) | 0.000 |
| IT Support | 654 | n/a (0/0) | 0.000 (0/654) | 0.000 |
| Product Support | 1020 | n/a (0/0) | 0.000 (0/1020) | 0.000 |
| Returns and Exchanges | 315 | n/a (0/0) | 0.000 (0/315) | 0.000 |
| Sales and Pre-Sales | 193 | n/a (0/0) | 0.000 (0/193) | 0.000 |
| Service Outages and Maintenance | 225 | n/a (0/0) | 0.000 (0/225) | 0.000 |
| Technical Support | 1708 | 0.299 (1708/5715) | 1.000 (1708/1708) | 0.460 |

Confusion matrix: [confusion_queue.png](confusion_queue.png), [confusion_queue.csv](confusion_queue.csv)

## Per-class: priority

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| high | 2294 | n/a (0/0) | 0.000 (0/2294) | 0.000 |
| low | 1124 | n/a (0/0) | 0.000 (0/1124) | 0.000 |
| medium | 2297 | 0.402 (2297/5715) | 1.000 (2297/2297) | 0.573 |

Confusion matrix: [confusion_priority.png](confusion_priority.png), [confusion_priority.csv](confusion_priority.csv)

## Outage events (type == Incident AND priority == high)

True events: 1047. Two flagging rules:

| rule | flagged | recall | precision |
|---|---|---|---|
| flag on pred Incident and pred high | 0 | 0.000 (0/1047) | n/a (0/0) |
| flag on pred high alone | 0 | 0.000 (0/1047) | n/a (0/0) |

- routing: no decision_status column

## By language detected from the body

langdetect on the holdout body, seed 0. Codes other than en and de, and empty bodies, are grouped as other.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3937 | 0.146 | 0.047 | 0.189 | 0.000 (0/722) | 0.000 (0/722) |
| de | 1770 | 0.140 | 0.044 | 0.196 | 0.000 (0/323) | 0.000 (0/323) |
| other | 8 | 0.256 | 0.080 | 0.182 | 0.000 (0/2) | 0.000 (0/2) |

## By language label

The dataset's `language` column, which is wrong for about 12% of rows.

| language | n | type macro-F1 | queue macro-F1 | priority macro-F1 | Incident+high recall, pred Incident and high | Incident+high recall, pred high alone |
|---|---|---|---|---|---|---|
| en | 3284 | 0.143 | 0.046 | 0.189 | 0.000 (0/594) | 0.000 (0/594) |
| de | 2431 | 0.146 | 0.046 | 0.194 | 0.000 (0/453) | 0.000 (0/453) |
| other | 0 | | | | | |
