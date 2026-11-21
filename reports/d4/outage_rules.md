# Outage rule variants (KB validation partition, provisional stand-in triage)

True Incident + high: 389 of 2286 (17.0%).

| variant | recall of true Incident + high | sent to on-call | precision |
|---|---|---|---|
| Predicted Incident + high (brief, rule 2) | 253/389 = 65.0% | 431 = 18.9% | 58.7% |
| Predicted high alone | 282/389 = 72.5% | 888 = 38.8% | 31.8% |
| Rule 2 + rule 6 at P(Incident) >= 0.35, P(high) >= 0.35 | 296/389 = 76.1% | 566 = 24.8% | 52.3% |
| Rule 2 + rule 6 + rule 2 wording | 296/389 = 76.1% | 567 = 24.8% | 52.2% |

Plan search for the rule 6 pair (0.30–0.70, step 0.05, highest precision with combined recall >= 90%): none reaches 90%.
D2 selects the release pair on its calibrated probabilities; this table shows the trade-off.
