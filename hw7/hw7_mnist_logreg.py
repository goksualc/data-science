from __future__ import annotations

import argparse
import gzip
import json
import struct
import time
import urllib.request
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

import certifi


BASE_URLS = [
    # Official MNIST host (sometimes blocked/refused on some networks)
    "https://yann.lecun.com/exdb/mnist/",
    # Common mirror (PyTorch datasets)
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download_from_any(fname: str, out_path: Path, *, sleep_s: float = 0.2) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    last_err: Exception | None = None
    context = ssl.create_default_context(cafile=certifi.where())
    for base in BASE_URLS:
        url = base + fname
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CISC7700X-HW7/0.1"})
            with urllib.request.urlopen(req, timeout=120, context=context) as resp:
                out_path.write_bytes(resp.read())
            time.sleep(sleep_s)
            return
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to download {fname} from all sources") from last_err
    time.sleep(sleep_s)


def _read_idx_images_gz(path: Path) -> np.ndarray:
    # IDX3: magic(2051), n, rows, cols, then uint8 pixels
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        magic, n, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != n * rows * cols:
        raise ValueError("Unexpected image data size")
    return arr.reshape(n, rows * cols)


def _read_idx_labels_gz(path: Path) -> np.ndarray:
    # IDX1: magic(2049), n, then uint8 labels
    with gzip.open(path, "rb") as f:
        header = f.read(8)
        magic, n = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic}")
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size != n:
        raise ValueError("Unexpected label data size")
    return arr.astype(int)


def load_mnist(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, fname in FILES.items():
        _download_from_any(fname, cache_dir / fname)

    X_train = _read_idx_images_gz(cache_dir / FILES["train_images"])
    y_train = _read_idx_labels_gz(cache_dir / FILES["train_labels"])
    X_test = _read_idx_images_gz(cache_dir / FILES["test_images"])
    y_test = _read_idx_labels_gz(cache_dir / FILES["test_labels"])

    return X_train, y_train, X_test, y_test


def main() -> int:
    ap = argparse.ArgumentParser(description="HW7: train HW6-style model on MNIST.")
    ap.add_argument("--cache-dir", type=str, default="hw7/.cache")
    ap.add_argument("--out-dir", type=str, default="hw7/outputs")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--solver", type=str, default="saga", help="saga is fast for large datasets.")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train_u8, y_train, X_test_u8, y_test = load_mnist(cache_dir)

    # Scale to [0,1] like typical MNIST baselines.
    X_train = X_train_u8.astype(np.float32) / 255.0
    X_test = X_test_u8.astype(np.float32) / 255.0

    # HW6-like linear classifier baseline.
    clf = LogisticRegression(
        solver=args.solver,
        max_iter=args.max_iter,
        C=args.C,
        random_state=args.seed,
        n_jobs=-1,
        verbose=0,
    )

    print("HW7 MNIST: LogisticRegression on scaled pixels")
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"solver={args.solver}  max_iter={args.max_iter}  C={args.C}  seed={args.seed}")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))

    print()
    print(f"Test accuracy: {acc:.6f}")
    print()
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    results = {
        "dataset": "MNIST (train/test from yann.lecun.com)",
        "model": "LogisticRegression (multiclass)",
        "solver": args.solver,
        "max_iter": args.max_iter,
        "C": args.C,
        "seed": args.seed,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / "hw7_mnist_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

