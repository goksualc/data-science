from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


@dataclass(frozen=True)
class TrainResult:
    model: Pipeline
    x_test: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray


def load_iris_from_sklearn() -> tuple[np.ndarray, np.ndarray]:
    iris = load_iris()
    x = iris.data.astype(float)
    y = np.array([iris.target_names[i] for i in iris.target], dtype=str)
    return x, y


def load_iris_from_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    x_rows: list[list[float]] = []
    y_rows: list[str] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FEATURES + [TARGET] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"Missing columns in CSV: {missing}. Found: {reader.fieldnames or []}"
            )

        for row in reader:
            x_rows.append([float(row[c]) for c in FEATURES])
            y_rows.append(str(row[TARGET]))

    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=str)


def train_knn(
    *,
    k: int,
    test_size: float,
    random_state: int,
    csv_path: Path | None,
) -> TrainResult:
    if k <= 0:
        raise ValueError("k must be >= 1")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1 (exclusive)")

    x, y = load_iris_from_csv(csv_path) if csv_path is not None else load_iris_from_sklearn()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return TrainResult(model=model, x_test=x_test, y_test=y_test, y_pred=y_pred)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="kNN classifier for Iris species from 4 measurements.",
    )
    p.add_argument("--k", type=int, default=5, help="Number of neighbors (default: 5)")
    p.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout fraction for evaluation (default: 0.2)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to a local iris.csv with header: "
        "sepal_length,sepal_width,petal_length,petal_width,species",
    )
    p.add_argument(
        "--predict",
        nargs=4,
        type=float,
        metavar=("SEPAL_LEN", "SEPAL_WID", "PETAL_LEN", "PETAL_WID"),
        help="If provided, prints predicted species for these 4 values.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv) if args.csv else None
    res = train_knn(k=args.k, test_size=args.test_size, random_state=args.seed, csv_path=csv_path)
    acc = accuracy_score(res.y_test, res.y_pred)
    print(f"Test accuracy: {acc:.4f}  (k={args.k}, test_size={args.test_size}, seed={args.seed})")
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(res.y_test, res.y_pred, labels=np.unique(res.y_test)))
    print()
    print("Classification report:")
    print(classification_report(res.y_test, res.y_pred))

    if args.predict is not None:
        x_one = np.array([args.predict], dtype=float)
        pred = res.model.predict(x_one)[0]
        probs = None
        if hasattr(res.model, "predict_proba"):
            probs = res.model.predict_proba(x_one)[0]
        print()
        print(f"Prediction for {dict(zip(FEATURES, args.predict))}: {pred}")
        if probs is not None:
            classes = list(res.model.named_steps["knn"].classes_)
            probs_dict = {cls: float(p) for cls, p in zip(classes, probs)}
            print(f"Probabilities: {probs_dict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
