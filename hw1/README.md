# CISC 7700X HW#1 — kNN on Iris

This repo trains a **k-Nearest Neighbors (kNN)** model to predict Iris flower species from:

- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`

Dataset source used by the script: `https://www.theparticle.com/cs/bc/dsci/iris.csv`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train + evaluate

```bash
python knn_iris.py
```

By default it uses the built-in Iris dataset from scikit-learn (so it runs offline).

Optional parameters:

```bash
python knn_iris.py --k 7 --test-size 0.25 --seed 1
```

## Use a local CSV (optional)

If you have a local `iris.csv` with columns:
`sepal_length,sepal_width,petal_length,petal_width,species`

```bash
python knn_iris.py --csv /path/to/iris.csv
```

## Predict a single flower

```bash
python knn_iris.py --predict 5.1 3.5 1.4 0.2
```

