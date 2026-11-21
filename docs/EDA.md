# Data exploration

Full numbers: [eda_stats.txt](eda_stats.txt). Figures: [figures/](figures/). Regenerate with `python scripts/eda.py`.

## Which file

Use only `aa_dataset-tickets-multi-lang-5-2-50-version.csv` (28,587 rows). The spec's `load_dataset(...)` snippet concatenates all three CSVs in the HF repo (61,765 rows), which breaks things:

- `dataset-tickets-multi-lang-4-20k.csv` repeats 8,395 bodies from the main file. A random split over the concatenation leaks test tickets into the KB.
- `dataset-tickets-german_normalized_50_5_2.csv` has 42 unrelated queues (e.g. "Pets & Animals/Pet Services"), 5 priority levels, and no `type` or `answer`.

## Locked split

`scripts/make_split.py`: fill null subject/answer with "", turn literal `\n` escapes into newlines (~4% of rows), drop 15 duplicate bodies, shuffle with seed 42, assign `TKT-00000..`, 80/20 split.

| file | rows | contents |
|---|---|---|
| `data/processed/kb_train.parquet` | 22,857 | everything, for training and RAG |
| `data/processed/test_inputs.parquet` | 5,715 | ticket_id, subject, body |
| `data/processed/test_labels.parquet` | 5,715 | ticket_id, type, queue, priority, language (eval only) |

Holdout answers are never written to disk. Train/test label shares differ by at most 1.3 points.

## Labels

- type: Incident 40%, Request 28.5%, Problem 21%, Change 10%
- priority: medium 40%, high 39%, low 21% (no "critical")
- queue: 10 classes, skewed. Technical Support 29% down to General Inquiry 1.4%. Expect the small queues to drag macro-F1.
- `priority==high & type==Incident` is 17% of tickets. That rule alone escalates about 1 in 6.
- Priority depends far more on queue than on type: Service Outages 70% high, Technical Support 58% high, Human Resources 9% high. Type barely moves it (Incident 42.5% high vs Request 33.5%).
- Label mix is the same in en and de.

## Text

- Bodies are short: median 54 words, p99 104. Answers median 58 words. Fits any encoder at 256 tokens.
- 13.5% of subjects are empty.
- `language` is noisy: about 26% of rows labelled `de` have an English body (langdetect, 3,000-row sample). Detect language from the text and reply in the detected language. Don't trust the column.
- Bodies open with heavy boilerplate ("Dear Customer Support Team, I hope this message finds you well..."). 511 test bodies share their first 80 characters with a KB ticket, but labels on those pairs mostly differ, so it's template text, not leakage. Strip greetings before embedding or retrieval will match on the greeting.
- Tags: 1,110 distinct, 4.7 per ticket. Top: Tech Support, IT, Performance, Feedback, Bug, Security.
- `version` column has values 400 / 52 / 51 (generator versions). Not a target.

## PII and placeholders

- Bodies have essentially no real emails, IPs or phone numbers. Data is synthetic and already masked.
- Answers are full of mask tokens: `<tel_num>` (8.4k), `<name>` (4.9k), `<acc_num>` (2k), `[Name]` (2k). The generator must not copy these into drafts, or should render them as "your account number" etc.
- Answers almost never mention money, percentages, refunds, guarantees or SLA hours (all under 0.5%). Any of those in a draft is unsupported by precedent, which makes a fabrication check simple.

## Baseline (TF-IDF word+char, LogisticRegression, class_weight=balanced)

| target | macro-F1 | accuracy | majority-class acc |
|---|---|---|---|
| type | 0.884 | 0.876 | 0.406 |
| queue | 0.587 | 0.575 | 0.299 |
| priority | 0.631 | 0.642 | 0.402 |

High-priority Incident recall 0.654 (n=1,047). Queue and priority are the hard targets.

## Environment note

On this Windows machine, Smart App Control blocks some fresh compiled wheels (Python 3.11 from uv, scikit-learn 1.9.1). Python 3.14 plus scikit-learn 1.8.0 works. Check torch / faiss / sentence-transformers early.
