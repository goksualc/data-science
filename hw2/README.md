# CISC 7700X HW#2 — Iris Histograms + Bootstrap CIs

This homework plots per-class histograms and computes summary statistics for:

- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`

For each label (`setosa`, `versicolor`, `virginica`), it computes:
- mean and standard deviation
- median and IQR (`Q3 - Q1`)

Then it uses a **percentile bootstrap** to estimate 95% confidence intervals (error bounds) for all of the above.

## Run

```bash
python3 iris_bootstrap_analysis.py
```

Common options:

```bash
python3 iris_bootstrap_analysis.py --n-bootstrap 20000 --alpha 0.05 --seed 123 --bins 15
```

## Inputs

- `hw2/iris.webarchive` (repo-provided data in CSV-like format)

## Outputs

- CSV summary with point estimates + bootstrap CIs:
  - `hw2/outputs/iris_bootstrap_summary.csv`
- Histograms:
  - `hw2/outputs/plots/sepal_length_hist.png`
  - `hw2/outputs/plots/sepal_width_hist.png`
  - `hw2/outputs/plots/petal_length_hist.png`
  - `hw2/outputs/plots/petal_width_hist.png`

