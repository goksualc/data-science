# CISC 7700X — Homework Repository

This repository contains solutions/scripts for **CISC 7700X** homeworks, organized by folder:

- `hw1/` — kNN on Iris
- `hw2/` — Iris histograms + bootstrap confidence intervals
- `hw3/` — cost-sensitive evaluation of a linear classifier
- `hw4/` — revenue/earnings/dividends modeling from SEC XBRL “companyfacts”
- `hw5/` — Naive Bayes spam classifier (Spambase)
- `hw6/` — classification model on `hw6.data.csv` (70/30 split)
- `hw7/` — run the HW6-style model on MNIST (train/test split)

Each homework folder contains its own `README.md` with:
- what the homework does
- how to run it
- where outputs are written

## Quick start

Most homeworks can be run from the repo root with `python3 ...` as shown in each folder README.

Examples:

```bash
python3 hw5/spambase_nb_train.py
python3 hw6/hw6_train_eval.py
python3 hw4/hw4_models.py
```

## Per-homework entrypoints

- **HW1**: `python3 hw1/knn_iris.py`
- **HW2**: `python3 hw2/iris_bootstrap_analysis.py`
- **HW3**: `python3 hw3/hw3_linear_model_eval.py`
- **HW4**: `python3 hw4/hw4_models.py`
- **HW5**: `python3 hw5/spambase_nb_train.py`
- **HW6**: `python3 hw6/hw6_train_eval.py`
- **HW7**: `python3 hw7/hw7_mnist_logreg.py`

