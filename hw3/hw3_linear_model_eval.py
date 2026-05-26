from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURES = [f"column{i}" for i in range(1, 21)]

# Coefficients from the prompt
COEFS: dict[str, float] = {
    "column1": 24.0,
    "column2": -15.0,
    "column3": -38.0,
    "column4": -7.0,
    "column5": -41.0,
    "column6": 35.0,
    "column7": 0.0,
    "column8": -2.0,
    "column9": 19.0,
    "column10": 33.0,
    "column11": -3.0,
    "column12": 7.0,
    "column13": 3.0,
    "column14": -47.0,
    "column15": 26.0,
    "column16": 10.0,
    "column17": 40.0,
    "column18": -1.0,
    "column19": 3.0,
    "column20": 0.0,
}
BIAS = -6.0


@dataclass(frozen=True)
class Confusion:
    tp: int
    tn: int
    fp: int
    fn: int


@dataclass(frozen=True)
class EvalResult:
    threshold: float
    accuracy: float
    confusion: Confusion
    cost_total: float
    economic_gain_total: float
    economic_gain_per_instance: float


def load_hw3_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: shape (N, 20), float
      y: shape (N,), int in {-1, +1}
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    x_rows: list[list[float]] = []
    y_rows: list[int] = []

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FEATURES + ["label"] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Found: {reader.fieldnames or []}")

        for row in reader:
            x_rows.append([float(row[c]) for c in FEATURES])
            y_rows.append(int(row["label"]))

    return np.asarray(x_rows, dtype=float), np.asarray(y_rows, dtype=int)


def scores_from_linear_model(X: np.ndarray) -> np.ndarray:
    # X columns correspond to column1..column20 in order
    w = np.asarray([COEFS[c] for c in FEATURES], dtype=float)
    return X @ w + BIAS


def compute_confusion(pred: np.ndarray, true: np.ndarray) -> Confusion:
    tp = int(np.sum((pred == 1) & (true == 1)))
    tn = int(np.sum((pred == -1) & (true == -1)))
    fp = int(np.sum((pred == 1) & (true == -1)))
    fn = int(np.sum((pred == -1) & (true == 1)))
    return Confusion(tp=tp, tn=tn, fp=fp, fn=fn)


def evaluate_threshold(scores: np.ndarray, true: np.ndarray, *, threshold: float, cost_fn: float, cost_fp: float) -> EvalResult:
    pred = np.where(scores > threshold, 1, -1)
    conf = compute_confusion(pred, true)
    n = true.shape[0]
    accuracy = (conf.tp + conf.tn) / n

    cost_total = cost_fn * conf.fn + cost_fp * conf.fp
    economic_gain_total = -cost_total  # correct = 0 cost
    economic_gain_per_instance = economic_gain_total / n

    return EvalResult(
        threshold=float(threshold),
        accuracy=float(accuracy),
        confusion=conf,
        cost_total=float(cost_total),
        economic_gain_total=float(economic_gain_total),
        economic_gain_per_instance=float(economic_gain_per_instance),
    )


def best_threshold_by_cost(scores: np.ndarray, true: np.ndarray, *, cost_fn: float, cost_fp: float) -> EvalResult:
    # Candidate thresholds at midpoints between sorted unique score values
    uniq = np.unique(scores)
    uniq_sorted = np.sort(uniq)
    candidates: list[float] = []
    for i in range(len(uniq_sorted) - 1):
        candidates.append((uniq_sorted[i] + uniq_sorted[i + 1]) / 2.0)
    candidates.append(float(uniq_sorted[0]) - 1.0)
    candidates.append(float(uniq_sorted[-1]) + 1.0)

    best: EvalResult | None = None
    for th in candidates:
        res = evaluate_threshold(scores, true, threshold=th, cost_fn=cost_fn, cost_fp=cost_fp)
        # maximize economic gain <=> minimize total cost
        if best is None or res.economic_gain_total > best.economic_gain_total:
            best = res
        elif res.economic_gain_total == best.economic_gain_total and res.accuracy > best.accuracy:
            best = res
    assert best is not None
    return best


def main() -> int:
    data_path = Path("hw3.data1.csv")  # run from hw3/ folder
    X, y = load_hw3_csv(data_path)
    scores = scores_from_linear_model(X)

    cost_fn = 1000.0
    cost_fp = 100.0

    # Your provided decision rule corresponds to threshold t=0
    res_default = evaluate_threshold(scores, y, threshold=0.0, cost_fn=cost_fn, cost_fp=cost_fp)
    res_best = best_threshold_by_cost(scores, y, cost_fn=cost_fn, cost_fp=cost_fp)

    out = {
        "N": int(y.shape[0]),
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "default_threshold": res_default.threshold,
        "default": {
            "accuracy": res_default.accuracy,
            "confusion": res_default.confusion.__dict__,
            "cost_total": res_default.cost_total,
            "economic_gain_total": res_default.economic_gain_total,
            "economic_gain_per_instance": res_default.economic_gain_per_instance,
        },
        "best_threshold": res_best.threshold,
        "best": {
            "accuracy": res_best.accuracy,
            "confusion": res_best.confusion.__dict__,
            "cost_total": res_best.cost_total,
            "economic_gain_total": res_best.economic_gain_total,
            "economic_gain_per_instance": res_best.economic_gain_per_instance,
        },
        # Equivalent intercept shift for the best threshold, using rule: predict +1 if score > t
        # score = linear + bias, so (score > t) == (linear + (bias - t) > 0)
        "best_equivalent_bias_shift": float(0.0 - res_best.threshold),  # since original uses + bias then compare to 0
    }

    print(json.dumps(out, indent=2))

    # Save for submission convenience
    out_path = Path("hw3_linear_model_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

