# A vs A-en on English val rows

n = 1330 val rows with language == 'en'. A: trained on all fit rows (en+de); A-en: trained on English fit rows only. Same features, C=4, temperatures refit on calib (A-en: English calib rows). Rule-6 thresholds re-searched on these rows for both.

| metric | A (en+de) | A-en | Δ |
|---|---|---|---|
| macro-F1 type | 0.893 | 0.883 | -0.009 |
| ECE type | 0.022 | 0.017 | -0.005 |
| macro-F1 queue | 0.609 | 0.607 | -0.002 |
| ECE queue | 0.032 | 0.031 | -0.001 |
| macro-F1 priority | 0.647 | 0.637 | -0.010 |
| ECE priority | 0.061 | 0.018 | -0.044 |
| Incident∧high recall | 0.692 | 0.679 | -0.013 |
| Incident∧high precision | 0.625 | 0.609 | -0.016 |
| rule 6 best recall | 0.846 | 0.821 | -0.026 |
| rule 6 escalation | 28.5% | 27.7% | -0.008 |
