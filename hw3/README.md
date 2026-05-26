# CISC 7700X HW#3 — Cost-Sensitive Linear Model

This homework evaluates a given linear model on `hw3.data1.csv` (repo copy of `hw3.data1.csv.gz`), using:
- confusion matrix / accuracy
- expected economic gain under asymmetric costs

The dataset columns are:
- features: `column1 ... column20`
- label: `label` in `{ -1, +1 }` (positive class is `+1`)

## Files

- `hw3_linear_model_eval.py`: computes confusion matrix/accuracy for the given coefficients, and finds a cost-optimized threshold by scanning over score thresholds.
- `hw3_linear_model_results.json`: script output (default + best).
- `hw3_linear_model_writeup.md`: the same results formatted for submission.

## Run

From the `hw3/` directory:

```bash
python3 hw3_linear_model_eval.py
```

It writes:
- `hw3_linear_model_results.json`

