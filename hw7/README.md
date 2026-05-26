# CISC 7700X HW#7 — HW6 model on MNIST

Goal: run the **same type of model used in HW6** on the MNIST digits dataset.

MNIST source (official):
- http://yann.lecun.com/exdb/mnist/

This solution downloads the 4 gz files, parses IDX format, trains on the **train** split,
and evaluates accuracy on the **test** split.

## Model

- Multiclass Logistic Regression (one-vs-rest handled internally as multinomial)
- Input scaling: `X / 255.0` to map pixels to `[0,1]`

## Run

From repo root:

```bash
python3 hw7/hw7_mnist_logreg.py
```

Outputs:
- prints test accuracy + confusion matrix
- writes `hw7/outputs/hw7_mnist_results.json`

