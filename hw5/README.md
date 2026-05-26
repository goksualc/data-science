# CISC 7700X HW#5 — Naive Bayes Spam Classifier (Spambase)

This homework builds a simple **Naive Bayes** email classifier using the UCI **Spambase** dataset.

- Dataset files live in `hw5/spambase/`:
  - `spambase.data` (CSV rows, **57 features + 1 label**)
  - `spambase.names` (feature descriptions)

This implementation uses **Gaussian Naive Bayes** (continuous features).

## Train + evaluate

From repo root:

```bash
python3 hw5/spambase_nb_train.py
```

Options:

```bash
python3 hw5/spambase_nb_train.py --test-size 0.2 --seed 42
```

Outputs:
- prints accuracy + confusion matrix
- writes model JSON: `hw5/model/spambase_gaussian_nb.json`

### Example result (seed=42, test_size=0.2)

- **Accuracy**: `0.8198`
- **Confusion matrix** (rows=true, cols=pred):
  - true=0: TN=406, FP=152
  - true=1: FN=14,  TP=349

## Classify

Classify a single CSV row of 57 comma-separated numbers (no label):

```bash
python3 hw5/spambase_nb_classify.py --row "0,0,0,...(57 values total)..."
```

Or classify all rows in a file (each row must have 57 feature values; label column is allowed but ignored):

```bash
python3 hw5/spambase_nb_classify.py --csv hw5/spambase/spambase.data
```

