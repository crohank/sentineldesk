# Triage variants on the holdout test set (D1 split)

Predictions: `scripts/predict_test.py` (reads only `test_inputs.parquet`). Scored once with D1's `scripts/run_eval.py` against `test_labels.parquet` rebuilt in a separate worktree of `origin/d1/data-eval` (all split SHA-256s match D1). The release model (SVM-tuned+extra) was fixed on val before scoring; test results did not feed back into any choice.

decision_status: on-call = val-chosen rule-6 thresholds (p_incident ≥ 0.3 and p_high ≥ 0.3); human_review = any head below confidence 0.45; else auto_draft. Per-variant detail in `reports/triage_d1/test/d2_<variant>*/`.

## full holdout (n = 5715)

| metric | A · TF-IDF + LR | A+extra · TF-IDF + LR +kb_extra | B · EmbeddingGemma + LR | SVM+extra · LinearSVC +kb_extra | SVM-tuned+extra · LinearSVC (tuned) +kb_extra (release) |
|---|---|---|---|---|---|
| type macro-F1 | 0.872 | 0.866 | 0.826 | 0.871 | 0.874 |
| queue macro-F1 | 0.558 | 0.545 | 0.325 | 0.591 | 0.605 |
| priority macro-F1 | 0.600 | 0.594 | 0.458 | 0.623 | 0.628 |
| type ECE, 10 bins | 0.003 | 0.002 | 0.007 | 0.010 | 0.011 |
| queue ECE, 10 bins | 0.021 | 0.021 | 0.038 | 0.022 | 0.029 |
| priority ECE, 10 bins | 0.020 | 0.016 | 0.011 | 0.022 | 0.030 |
| Incident+high recall, flag on pred Incident and pred high | 0.625 | 0.633 | 0.516 | 0.670 | 0.655 |
| Incident+high precision, flag on pred Incident and pred high | 0.608 | 0.617 | 0.520 | 0.623 | 0.632 |
| true Incident+high routed to on-call | 0.793 | 0.796 | 0.757 | 0.781 | 0.778 |
| overall on-call rate | 0.285 | 0.281 | 0.324 | 0.272 | 0.265 |
| true Incident+high routed to human | 0.904 | 0.918 | 0.971 | 0.897 | 0.882 |
| overall escalation rate | 0.652 | 0.667 | 0.928 | 0.606 | 0.580 |

## no near-dups (n = 5703)

| metric | A · TF-IDF + LR | A+extra · TF-IDF + LR +kb_extra | B · EmbeddingGemma + LR | SVM+extra · LinearSVC +kb_extra | SVM-tuned+extra · LinearSVC (tuned) +kb_extra (release) |
|---|---|---|---|---|---|
| type macro-F1 | 0.872 | 0.866 | 0.825 | 0.871 | 0.874 |
| queue macro-F1 | 0.558 | 0.545 | 0.326 | 0.591 | 0.604 |
| priority macro-F1 | 0.600 | 0.594 | 0.457 | 0.623 | 0.627 |
| type ECE, 10 bins | 0.003 | 0.002 | 0.007 | 0.010 | 0.011 |
| queue ECE, 10 bins | 0.022 | 0.022 | 0.038 | 0.022 | 0.029 |
| priority ECE, 10 bins | 0.021 | 0.016 | 0.010 | 0.021 | 0.030 |
| Incident+high recall, flag on pred Incident and pred high | 0.624 | 0.633 | 0.515 | 0.670 | 0.655 |
| Incident+high precision, flag on pred Incident and pred high | 0.607 | 0.616 | 0.519 | 0.622 | 0.631 |
| true Incident+high routed to on-call | 0.792 | 0.795 | 0.757 | 0.781 | 0.778 |
| overall on-call rate | 0.285 | 0.282 | 0.324 | 0.272 | 0.265 |
| true Incident+high routed to human | 0.904 | 0.918 | 0.971 | 0.897 | 0.881 |
| overall escalation rate | 0.652 | 0.667 | 0.928 | 0.606 | 0.581 |

## no twins (n = 5195)

| metric | A · TF-IDF + LR | A+extra · TF-IDF + LR +kb_extra | B · EmbeddingGemma + LR | SVM+extra · LinearSVC +kb_extra | SVM-tuned+extra · LinearSVC (tuned) +kb_extra (release) |
|---|---|---|---|---|---|
| type macro-F1 | 0.864 | 0.859 | 0.824 | 0.863 | 0.865 |
| queue macro-F1 | 0.529 | 0.516 | 0.321 | 0.559 | 0.570 |
| priority macro-F1 | 0.575 | 0.568 | 0.453 | 0.597 | 0.599 |
| type ECE, 10 bins | 0.005 | 0.006 | 0.009 | 0.015 | 0.015 |
| queue ECE, 10 bins | 0.017 | 0.018 | 0.039 | 0.017 | 0.026 |
| priority ECE, 10 bins | 0.013 | 0.025 | 0.011 | 0.014 | 0.021 |
| Incident+high recall, flag on pred Incident and pred high | 0.612 | 0.618 | 0.511 | 0.655 | 0.637 |
| Incident+high precision, flag on pred Incident and pred high | 0.590 | 0.600 | 0.518 | 0.606 | 0.611 |
| true Incident+high routed to on-call | 0.781 | 0.785 | 0.750 | 0.770 | 0.765 |
| overall on-call rate | 0.293 | 0.288 | 0.328 | 0.279 | 0.273 |
| true Incident+high routed to human | 0.900 | 0.913 | 0.970 | 0.892 | 0.876 |
| overall escalation rate | 0.674 | 0.687 | 0.931 | 0.631 | 0.609 |
