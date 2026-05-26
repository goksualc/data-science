from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODEL_PATH_DEFAULT = Path("hw5/model/spambase_gaussian_nb.json")


def load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def parse_row(row: str) -> np.ndarray:
    parts = [p.strip() for p in row.strip().split(",") if p.strip() != ""]
    if len(parts) != 57:
        raise ValueError(f"--row must have exactly 57 comma-separated values, got {len(parts)}")
    return np.asarray([float(p) for p in parts], dtype=float)


def load_rows_from_csv(path: Path) -> np.ndarray:
    """
    Accepts:
    - 57 columns (features only), or
    - 58 columns (features + label) where the label is ignored.
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
            if len(parts) == 57:
                rows.append([float(p) for p in parts])
            elif len(parts) == 58:
                rows.append([float(p) for p in parts[:57]])
            else:
                raise ValueError(f"Expected 57 or 58 columns, got {len(parts)}")

    return np.asarray(rows, dtype=float)


def predict_log_posteriors(model: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    means = np.asarray(model["means"], dtype=float)  # (C,F)
    vars_ = np.asarray(model["vars"], dtype=float)   # (C,F)
    priors = np.asarray(model["priors"], dtype=float)  # (C,)

    const = -0.5 * np.log(2.0 * np.pi * vars_)  # (C,F)
    quad = -((X[:, None, :] - means[None, :, :]) ** 2) / (2.0 * vars_[None, :, :])
    log_lik = (const[None, :, :] + quad).sum(axis=2)  # (N,C)
    return log_lik + np.log(priors)[None, :]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    m = np.max(logits, axis=1, keepdims=True)
    ex = np.exp(logits - m)
    return ex / np.sum(ex, axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify using a trained Spambase Gaussian NB model.")
    ap.add_argument("--model", type=str, default=str(MODEL_PATH_DEFAULT))
    ap.add_argument("--row", type=str, default=None, help="57 comma-separated feature values (no label).")
    ap.add_argument("--csv", type=str, default=None, help="CSV file with 57 features (or 57+label).")
    args = ap.parse_args()

    model = load_model(Path(args.model))
    classes = np.asarray(model["classes"], dtype=int)

    if (args.row is None) == (args.csv is None):
        raise ValueError("Provide exactly one of --row or --csv.")

    if args.row is not None:
        x = parse_row(args.row)[None, :]
        log_post = predict_log_posteriors(model, x)
        probs = softmax(log_post)[0]
        pred = int(classes[int(np.argmax(log_post[0]))])
        # label 1 = spam, 0 = non-spam
        print(f"predicted_label={pred}  p0={probs[classes.tolist().index(0)]:.6f}  p1={probs[classes.tolist().index(1)]:.6f}")
        return 0

    X = load_rows_from_csv(Path(args.csv))
    log_post = predict_log_posteriors(model, X)
    pred_idx = np.argmax(log_post, axis=1)
    preds = classes[pred_idx]
    # Print one label per line
    for p in preds.tolist():
        print(int(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

