from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH_DEFAULT = Path("hw6/hw6.data.csv")


def load_hw6_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads hw6.data.csv:
      - numeric columns
      - last column is the label (integer)
    Returns:
      X: (N, D) float
      y: (N,) int
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    x_rows: list[list[float]] = []
    y_rows: list[int] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            *feat, label = row
            x_rows.append([float(v) for v in feat])
            y_rows.append(int(float(label)))

    X = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=int)
    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("Unexpected data shape.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("Row count mismatch between X and y.")
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser(description="HW6: train/test classification model (70/30 split).")
    ap.add_argument("--data", type=str, default=str(DATA_PATH_DEFAULT))
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="hw6/outputs")
    args = ap.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_hw6_csv(data_path)
    labels = np.unique(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    # Model: scaled multinomial logistic regression (strong baseline for numeric features).
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=None,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("HW6 classifier: StandardScaler + LogisticRegression(lbfgs)")
    print(f"Data: {data_path}")
    print(f"N={X.shape[0]}  D={X.shape[1]}  labels={labels.tolist()}")
    print(f"Split: train={X_train.shape[0]} ({1.0-args.test_size:.0%})  test={X_test.shape[0]} ({args.test_size:.0%})  seed={args.seed}")
    print()
    print(f"Test accuracy: {acc:.6f}")
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    results = {
        "data": str(data_path),
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "labels": labels.tolist(),
        "test_size": float(args.test_size),
        "seed": int(args.seed),
        "model": "StandardScaler + LogisticRegression(lbfgs)",
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / "hw6_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

