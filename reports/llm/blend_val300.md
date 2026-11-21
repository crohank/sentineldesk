# Blend test: TF-IDF + Gemini on 300 val tickets

Same 300 `val` tickets as `val300_comparison.md`. TF-IDF fit on `fit`. Blend per head: w x Gemini + (1 - w) x TF-IDF. **Pre-registered w = 0.5**; other rows are context only and must not be used to pick w (that happens on `calib`).

## val300_flash_few8

| w (Gemini share) | type F1 | queue F1 | priority F1 | type ECE | queue ECE | priority ECE | Incident+high recall | precision |
|---|---|---|---|---|---|---|---|---|
| 0.0 (TF-IDF only) | 0.890 | 0.570 | 0.598 | 0.049 | 0.082 | 0.033 | 0.627 (32/51) | 0.552 (32/58) |
| 0.25 | 0.910 | 0.691 | 0.683 | 0.074 | 0.182 | 0.079 | 0.706 (36/51) | 0.621 (36/58) |
| 0.5 (pre-registered) | 0.939 | 0.759 | 0.764 | 0.110 | 0.189 | 0.134 | 0.902 (46/51) | 0.687 (46/67) |
| 0.75 | 0.955 | 0.750 | 0.760 | 0.114 | 0.173 | 0.101 | 0.863 (44/51) | 0.721 (44/61) |
| 1.0 (Gemini probs only) | 0.950 | 0.754 | 0.766 | 0.097 | 0.106 | 0.054 | 0.863 (44/51) | 0.759 (44/58) |

Gemini's stated label (not argmax of its probabilities): type 0.950, queue 0.754, priority 0.766.

Agreement between TF-IDF and Gemini argmax, and accuracy of the 50/50 blend on each part:

| head | agree | blend acc when agree | blend acc when disagree |
|---|---|---|---|
| type | 0.883 (265/300) | 0.970 | 0.657 |
| queue | 0.563 (169/300) | 0.834 | 0.588 |
| priority | 0.603 (181/300) | 0.834 | 0.672 |

## val300_lite_few8

| w (Gemini share) | type F1 | queue F1 | priority F1 | type ECE | queue ECE | priority ECE | Incident+high recall | precision |
|---|---|---|---|---|---|---|---|---|
| 0.0 (TF-IDF only) | 0.890 | 0.570 | 0.598 | 0.049 | 0.082 | 0.033 | 0.627 (32/51) | 0.552 (32/58) |
| 0.25 | 0.903 | 0.656 | 0.649 | 0.086 | 0.167 | 0.065 | 0.647 (33/51) | 0.559 (33/59) |
| 0.5 (pre-registered) | 0.922 | 0.682 | 0.712 | 0.130 | 0.183 | 0.129 | 0.804 (41/51) | 0.621 (41/66) |
| 0.75 | 0.922 | 0.665 | 0.714 | 0.143 | 0.140 | 0.112 | 0.784 (40/51) | 0.625 (40/64) |
| 1.0 (Gemini probs only) | 0.917 | 0.669 | 0.709 | 0.146 | 0.105 | 0.080 | 0.765 (39/51) | 0.639 (39/61) |

Gemini's stated label (not argmax of its probabilities): type 0.918, queue 0.681, priority 0.709.

Agreement between TF-IDF and Gemini argmax, and accuracy of the 50/50 blend on each part:

| head | agree | blend acc when agree | blend acc when disagree |
|---|---|---|---|
| type | 0.880 (264/300) | 0.951 | 0.667 |
| queue | 0.560 (168/300) | 0.786 | 0.485 |
| priority | 0.597 (179/300) | 0.793 | 0.603 |
