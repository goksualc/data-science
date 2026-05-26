from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABELS = ["setosa", "versicolor", "virginica"]


def load_iris_webarchive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads this repo's hw2/iris.webarchive format:
    sepal_length,sepal_width,petal_length,petal_width,species
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    x_rows: list[list[float]] = []
    y_rows: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                raise ValueError(f"Unexpected number of columns ({len(parts)}): {line}")
            x_rows.append([float(parts[i]) for i in range(4)])
            y_rows.append(parts[4])

    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=str)
    return x, y


@dataclass(frozen=True)
class Ci:
    low: float
    high: float


def percentile_ci(values: np.ndarray, *, alpha: float) -> Ci:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1 (exclusive)")
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    return Ci(low=float(np.quantile(values, lower_q)), high=float(np.quantile(values, upper_q)))


def bootstrap_stats_for_vector(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> dict[str, tuple[float, Ci]]:
    """
    Returns bootstrap percentile CIs for:
    - mean
    - sample std (ddof=1)
    - median
    - IQR (Q3-Q1, using NumPy's default linear quantile interpolation)
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples to estimate std/IQR.")

    rng = np.random.default_rng(seed)

    # Shape: (n_bootstrap, n)
    boot = values[rng.integers(0, n, size=(n_bootstrap, n))]

    mean_boot = boot.mean(axis=1)
    std_boot = boot.std(axis=1, ddof=1)
    median_boot = np.median(boot, axis=1)
    q1_boot = np.quantile(boot, 0.25, axis=1)
    q3_boot = np.quantile(boot, 0.75, axis=1)
    iqr_boot = q3_boot - q1_boot

    mean_point = float(values.mean())
    std_point = float(values.std(ddof=1))
    median_point = float(np.median(values))
    q1_point = float(np.quantile(values, 0.25))
    q3_point = float(np.quantile(values, 0.75))
    iqr_point = q3_point - q1_point

    return {
        "mean": (mean_point, percentile_ci(mean_boot, alpha=alpha)),
        "std": (std_point, percentile_ci(std_boot, alpha=alpha)),
        "median": (median_point, percentile_ci(median_boot, alpha=alpha)),
        "iqr": (iqr_point, percentile_ci(iqr_boot, alpha=alpha)),
    }


def plot_histograms(
    x: np.ndarray,
    y: np.ndarray,
    *,
    output_dir: Path,
    bins: int,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    label_to_color = {
        "setosa": "#1f77b4",
        "versicolor": "#ff7f0e",
        "virginica": "#2ca02c",
    }

    for feat_idx, feat_name in enumerate(FEATURES):
        fig, ax = plt.subplots(figsize=(8.5, 5.0))

        values_all = x[:, feat_idx]
        bin_edges = np.histogram_bin_edges(values_all, bins=bins)

        for label in LABELS:
            mask = y == label
            ax.hist(
                x[mask, feat_idx],
                bins=bin_edges,
                alpha=0.45,
                density=False,
                color=label_to_color.get(label, None),
                label=label,
            )

        ax.set_title(f"Iris histograms: {feat_name}")
        ax.set_xlabel(feat_name)
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)

        out_path = output_dir / f"{feat_name}_hist.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Iris bootstrap analysis (mean/std/median/IQR).")
    p.add_argument("--iris-webarchive", type=str, default="hw2/iris.webarchive")
    p.add_argument("--output-dir", type=str, default="hw2/outputs")
    p.add_argument("--n-bootstrap", type=int, default=20000)
    p.add_argument("--alpha", type=float, default=0.05, help="CI error rate; 0.05 => 95% CI.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--bins", type=int, default=15, help="Histogram bin count per feature.")
    args = p.parse_args()

    iris_path = Path(args.iris_webarchive)
    out_dir = Path(args.output_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y = load_iris_webarchive(iris_path)

    # Sanity check: ensure labels exist (and in expected order).
    for label in LABELS:
        n_label = int(np.sum(y == label))
        if n_label == 0:
            raise ValueError(f"Missing label in dataset: {label}")

    rows: list[dict[str, object]] = []
    for feat_idx, feat_name in enumerate(FEATURES):
        for label in LABELS:
            mask = y == label
            values = x[mask, feat_idx]

            # Vary bootstrap seed by (label, feature) for reproducibility.
            seed = args.seed + feat_idx * 100 + LABELS.index(label)

            stats = bootstrap_stats_for_vector(
                values,
                n_bootstrap=args.n_bootstrap,
                alpha=args.alpha,
                seed=seed,
            )

            n = int(values.shape[0])
            mean_point, mean_ci = stats["mean"]
            std_point, std_ci = stats["std"]
            median_point, median_ci = stats["median"]
            iqr_point, iqr_ci = stats["iqr"]

            rows.append(
                {
                    "attribute": feat_name,
                    "label": label,
                    "n": n,
                    "mean": mean_point,
                    "mean_ci_low": mean_ci.low,
                    "mean_ci_high": mean_ci.high,
                    "std": std_point,
                    "std_ci_low": std_ci.low,
                    "std_ci_high": std_ci.high,
                    "median": median_point,
                    "median_ci_low": median_ci.low,
                    "median_ci_high": median_ci.high,
                    "iqr": iqr_point,
                    "iqr_ci_low": iqr_ci.low,
                    "iqr_ci_high": iqr_ci.high,
                }
            )

    out_csv = out_dir / "iris_bootstrap_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print a compact summary for each attribute.
    print(f"Loaded {x.shape[0]} rows from {iris_path}")
    print(f"Bootstrap: n_bootstrap={args.n_bootstrap}, alpha={args.alpha} (CI level={1.0-args.alpha:.2f})")
    print()
    for feat_name in FEATURES:
        print(f"== {feat_name} ==")
        for label in LABELS:
            r = next(rr for rr in rows if rr["attribute"] == feat_name and rr["label"] == label)
            print(
                f"{label:>10}: "
                f"mean={r['mean']:.4f} [{r['mean_ci_low']:.4f}, {r['mean_ci_high']:.4f}]  "
                f"std={r['std']:.4f} [{r['std_ci_low']:.4f}, {r['std_ci_high']:.4f}]  "
                f"median={r['median']:.4f} [{r['median_ci_low']:.4f}, {r['median_ci_high']:.4f}]  "
                f"iqr={r['iqr']:.4f} [{r['iqr_ci_low']:.4f}, {r['iqr_ci_high']:.4f}]"
            )
        print()

    plot_histograms(x, y, output_dir=plots_dir, bins=args.bins)
    print(f"Plots saved to: {plots_dir}")
    print(f"CSV saved to: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

