from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_PATH_DEFAULT = Path("hw5/spambase/spambase.data")
MODEL_PATH_DEFAULT = Path("hw5/model/spambase_gaussian_nb.json")


@dataclass(frozen=True)
class Confusion:
    tp: int
    tn: int
    fp: int
    fn: int


def load_spambase(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    spambase.data format: 57 continuous features, then label in {0,1} (spam=1).
    Returns X (N,57) float, y (N,) int in {0,1}.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 58:
                raise ValueError(f"Expected 58 columns (57 + label), got {len(parts)}")
            rows.append([float(p) for p in parts])

    arr = np.asarray(rows, dtype=float)
    X = arr[:, :57]
    y = arr[:, 57].astype(int)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels must be 0/1.")
    return X, y


def stratified_split(X: np.ndarray, y: np.ndarray, *, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0,1)")
    rng = np.random.default_rng(seed)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0_test = int(round(test_size * len(idx0)))
    n1_test = int(round(test_size * len(idx1)))

    test_idx = np.concatenate([idx0[:n0_test], idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:], idx1[n1_test:]])
    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def fit_gaussian_nb(X: np.ndarray, y: np.ndarray, *, eps: float = 1e-9) -> dict:
    """
    Gaussian Naive Bayes:
      p(x_j | class=c) ~ Normal(mean[c,j], var[c,j])
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or X.shape[1] != 57:
        raise ValueError("Expected X shape (N,57)")

    classes = [0, 1]
    means = []
    vars_ = []
    priors = []

    for c in classes:
        Xc = X[y == c]
        if Xc.shape[0] == 0:
            raise ValueError(f"No samples for class {c}")
        means.append(Xc.mean(axis=0))
        # Use ML variance (ddof=0); add eps to avoid divide-by-zero
        vars_.append(Xc.var(axis=0) + eps)
        priors.append(Xc.shape[0] / X.shape[0])

    model = {
        "type": "gaussian_nb",
        "feature_count": 57,
        "classes": classes,
        "means": np.stack(means, axis=0).tolist(),  # (2,57)
        "vars": np.stack(vars_, axis=0).tolist(),    # (2,57)
        "priors": priors,                            # (2,)
        "eps": eps,
    }
    return model


def predict_gaussian_nb(model: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    means = np.asarray(model["means"], dtype=float)  # (C,F)
    vars_ = np.asarray(model["vars"], dtype=float)   # (C,F)
    priors = np.asarray(model["priors"], dtype=float)  # (C,)

    # log p(c) + sum_j log N(x_j; mean, var)
    # log N = -0.5*log(2*pi*var) - (x-mean)^2/(2*var)
    const = -0.5 * np.log(2.0 * np.pi * vars_)  # (C,F)
    # broadcast X: (N,1,F)
    quad = -((X[:, None, :] - means[None, :, :]) ** 2) / (2.0 * vars_[None, :, :])
    log_lik = (const[None, :, :] + quad).sum(axis=2)  # (N,C)
    log_post = log_lik + np.log(priors)[None, :]
    # class 1 if argmax is index of class=1
    preds_idx = np.argmax(log_post, axis=1)
    classes = np.asarray(model["classes"], dtype=int)
    return classes[preds_idx]


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Confusion:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return Confusion(tp=tp, tn=tn, fp=fp, fn=fn)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Gaussian Naive Bayes on Spambase.")
    ap.add_argument("--data", type=str, default=str(DATA_PATH_DEFAULT))
    ap.add_argument("--model-out", type=str, default=str(MODEL_PATH_DEFAULT))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=1e-9)
    args = ap.parse_args()

    data_path = Path(args.data)
    model_out = Path(args.model_out)

    X, y = load_spambase(data_path)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=args.test_size, seed=args.seed)

    model = fit_gaussian_nb(X_train, y_train, eps=args.eps)
    y_pred = predict_gaussian_nb(model, X_test)

    acc = float(np.mean(y_pred == y_test))
    conf = confusion_matrix_binary(y_test, y_pred)

    print(f"Spambase Gaussian NB (test_size={args.test_size}, seed={args.seed})")
    print(f"Train: N={X_train.shape[0]}   Test: N={X_test.shape[0]}")
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    print("           pred=0    pred=1")
    print(f"true=0     {conf.tn:7d}  {conf.fp:7d}")
    print(f"true=1     {conf.fn:7d}  {conf.tp:7d}")
    print()
    print(f"Accuracy: {acc:.4f}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Saved model: {model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

