from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SYMBOLS = ["IBM", "MSFT", "AAPL", "GOOG", "META", "PG", "GE"]


REV_TAG_CANDIDATES = [
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomer",
]

EARN_TAG_CANDIDATES = [
    "NetIncomeLoss",
    "IncomeLossFromContinuingOperations",
]

DIV_TAG_CANDIDATES = [
    # Prefer per-share dividends (more comparable across companies).
    "CommonStockDividendsPerShareDeclared",
    "DividendsPerShareCommonStock",
    "CommonStockDividendsPerShareCash",
    # Alternative per-share tag variants
    "DividendsPerShareCommonStockCash",
    # Fallback: total cash dividends / payments
    "PaymentsOfOrdinaryDividends",
    "PaymentsOfDividends",
    "PaymentsOfDividendsCommonStock",
    "CashDividendsPaidToParentCompany",
    "CashDividendsPaidToParentCompanyByConsolidatedSubsidiaries",
    "DividendsPaid",
]

QUARTER_FPS = {"Q1", "Q2", "Q3", "Q4"}


def sec_headers() -> dict[str, str]:
    # SEC requests a descriptive User-Agent. Put your real email/contact here if required by your course.
    return {
        "User-Agent": "CISC7700X-HW4/0.1 (contact: student@example.com)",
        # Avoid gzip/deflate so we can reliably decode JSON.
        "Accept-Encoding": "identity",
        "Accept": "application/json",
    }


def fetch_json(url: str, *, cache_path: Path | None = None, sleep_s: float = 0.2) -> dict[str, Any]:
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    import ssl
    import urllib.request

    # Fix TLS cert verification issues on some macOS installs by explicitly using certifi's CA bundle.
    import certifi

    context = ssl.create_default_context(cafile=certifi.where())

    req = urllib.request.Request(url, headers=sec_headers())
    with urllib.request.urlopen(req, timeout=60, context=context) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data), encoding="utf-8")

    time.sleep(sleep_s)
    return data


def load_ticker_to_cik() -> dict[str, str]:
    # From SEC: https://www.sec.gov/files/company_tickers.json
    url = "https://www.sec.gov/files/company_tickers.json"
    data = fetch_json(url, cache_path=Path("hw4/.cache/company_tickers.json"), sleep_s=0.4)
    mapping: dict[str, str] = {}
    for item in data.values():
        ticker = (item.get("ticker") or "").upper()
        cik = str(item.get("cik_str") or "").strip()
        if ticker and cik:
            mapping[ticker] = cik.lstrip("0") or cik
    return mapping


def cik_url(cik: str) -> str:
    # SEC companyfacts endpoint uses zero-padded CIK (e.g. CIK0000320193.json).
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"


def fetch_companyfacts(cik: str, *, cache_dir: Path) -> dict[str, Any]:
    cache_path = cache_dir / f"companyfacts_CIK{int(cik):d}.json"
    return fetch_json(cik_url(cik), cache_path=cache_path, sleep_s=0.3)


def parse_end_date(s: str) -> datetime:
    # Example: "2023-03-31"
    return datetime.strptime(s, "%Y-%m-%d")


@dataclass(frozen=True)
class QuarterlyPoint:
    end: datetime
    fp: str
    val: float


def extract_quarterly_series_from_tag(
    companyfacts: dict[str, Any],
    *,
    tag: str,
    min_points: int,
) -> list[QuarterlyPoint]:
    # Shape: facts -> us-gaap -> <tag> -> units -> <unit> -> list[entry]
    facts = companyfacts.get("facts", {})
    us_gaap = facts.get("us-gaap", {})
    tag_obj = us_gaap.get(tag, {})
    units = tag_obj.get("units", {})
    if not isinstance(units, dict) or not units:
        return []

    best: list[QuarterlyPoint] = []
    for _unit_name, entries in units.items():
        if not isinstance(entries, list):
            continue

        # Group by "end" (period end). If multiple entries exist for the same end, take the last by filed date.
        by_end: dict[str, tuple[datetime, str, float, str]] = {}
        for e in entries:
            try:
                fp = e.get("fp")
                if fp not in QUARTER_FPS:
                    continue
                end = e.get("end")
                if not end:
                    continue
                val = e.get("val")
                if val is None:
                    continue
                filed = e.get("filed") or ""  # some entries might omit filed
                end_dt = parse_end_date(end)
                # Keep last filed for that end date.
                prev = by_end.get(end)
                if prev is None:
                    by_end[end] = (end_dt, fp, float(val), filed)
                else:
                    if filed >= prev[2]:
                        by_end[end] = (end_dt, fp, float(val), filed)
            except Exception:
                # Be resilient to malformed entries.
                continue

        series = [QuarterlyPoint(end=dt, fp=fp, val=val) for (dt, fp, val, _filed) in by_end.values()]
        series.sort(key=lambda p: p.end)
        if len(series) > len(best):
            best = series

    # Return the full quarterly series. Caller decides how much to train on.
    return best


def extract_best_quarterly_series(
    companyfacts: dict[str, Any],
    *,
    tag_candidates: list[str],
    min_points: int,
) -> tuple[str | None, list[QuarterlyPoint]]:
    best_tag: str | None = None
    best_series: list[QuarterlyPoint] = []
    best_last_end: datetime | None = None

    # Selection rule:
    # - Prefer tags with at least min_points points (so we can make the required holdout prediction).
    # - Among those, choose the tag whose last available quarter end date is the most recent.
    # - If there's still a tie, prefer the tag with more total points.

    for tag in tag_candidates:
        series = sorted(extract_quarterly_series_from_tag(companyfacts, tag=tag, min_points=min_points), key=lambda p: p.end)
        if not series:
            continue

        meets = len(series) >= min_points
        last_end = series[-1].end

        if best_last_end is None:
            best_tag = tag
            best_series = series
            best_last_end = last_end if meets else None
            continue

        # If we have a current best that meets min_points, only replace if this one also meets and is more recent.
        if best_last_end is not None:
            if meets and last_end > best_last_end:
                best_tag = tag
                best_series = series
                best_last_end = last_end
            elif meets and last_end == best_last_end and len(series) > len(best_series):
                best_tag = tag
                best_series = series
        else:
            # Current best doesn't meet min_points: replace if this one meets, or if this one has more points.
            if meets:
                best_tag = tag
                best_series = series
                best_last_end = last_end
            elif len(series) > len(best_series):
                best_tag = tag
                best_series = series

    return best_tag, best_series


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class ModelFit:
    model_type: str
    params: dict[str, float]
    y_pred_train: np.ndarray
    r2_train: float


def fit_linear(x: np.ndarray, y: np.ndarray) -> ModelFit:
    # y = a + b x
    X = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    y_pred = a + b * x
    return ModelFit(model_type="linear", params={"a": a, "b": b}, y_pred_train=y_pred, r2_train=r2_score(y, y_pred))


def fit_log(x: np.ndarray, y: np.ndarray) -> ModelFit:
    # y = a + b log(x)
    lx = np.log(x)
    X = np.vstack([np.ones_like(lx), lx]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(coef[0])
    b = float(coef[1])
    y_pred = a + b * lx
    return ModelFit(model_type="logarithmic", params={"a": a, "b": b}, y_pred_train=y_pred, r2_train=r2_score(y, y_pred))


def fit_exponential(x: np.ndarray, y: np.ndarray) -> ModelFit | None:
    # y = b * exp(a*x)
    if np.any(y <= 0):
        return None
    ly = np.log(y)
    X = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    A = float(coef[0])  # log(b)
    a = float(coef[1])
    b = float(np.exp(A))
    y_pred = b * np.exp(a * x)
    return ModelFit(
        model_type="exponential",
        params={"a": a, "b": b},
        y_pred_train=y_pred,
        r2_train=r2_score(y, y_pred),
    )


def fit_power(x: np.ndarray, y: np.ndarray) -> ModelFit | None:
    # y = b * x^a
    if np.any(y <= 0) or np.any(x <= 0):
        return None
    lx = np.log(x)
    ly = np.log(y)
    X = np.vstack([np.ones_like(lx), lx]).T
    coef, *_ = np.linalg.lstsq(X, ly, rcond=None)
    logb = float(coef[0])
    a = float(coef[1])
    b = float(np.exp(logb))
    y_pred = b * (x**a)
    return ModelFit(
        model_type="power",
        params={"a": a, "b": b},
        y_pred_train=y_pred,
        r2_train=r2_score(y, y_pred),
    )


def fit_all_models(x: np.ndarray, y: np.ndarray) -> list[ModelFit]:
    fits: list[ModelFit] = []
    fits.append(fit_linear(x, y))
    fits.append(fit_log(x, y))
    exp_fit = fit_exponential(x, y)
    if exp_fit is not None:
        fits.append(exp_fit)
    pow_fit = fit_power(x, y)
    if pow_fit is not None:
        fits.append(pow_fit)
    return fits


def predict_from_fit(fit: ModelFit, x: float) -> float:
    t = fit.model_type
    if t == "linear":
        return fit.params["a"] + fit.params["b"] * x
    if t == "logarithmic":
        return fit.params["a"] + fit.params["b"] * math.log(x)
    if t == "exponential":
        return fit.params["b"] * math.exp(fit.params["a"] * x)
    if t == "power":
        return fit.params["b"] * (x**fit.params["a"])
    raise ValueError(f"Unknown model_type: {t}")


def main() -> int:
    ap = argparse.ArgumentParser(description="HW4: quarterly revenue/earnings/dividends modeling via SEC companyfacts.")
    ap.add_argument("--out-dir", type=str, default="hw4/outputs")
    ap.add_argument("--n-train", type=int, default=8, help="Train on previous N quarters; hold out the latest for prediction.")
    ap.add_argument("--min-points", type=int, default=9, help="Need at least N+1 points (latest held-out) to do the task.")
    ap.add_argument("--symbols", type=str, default=",".join(SYMBOLS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path("hw4/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    ticker_to_cik = load_ticker_to_cik()

    metrics = [
        ("revenue", REV_TAG_CANDIDATES),
        ("earnings", EARN_TAG_CANDIDATES),
        ("dividends", DIV_TAG_CANDIDATES),
    ]

    rows: list[dict[str, Any]] = []

    for sym in symbols:
        if sym not in ticker_to_cik:
            print(f"[WARN] Missing CIK for {sym}; skipping.")
            continue
        cik = ticker_to_cik[sym]
        print(f"[INFO] Fetching companyfacts for {sym} (CIK={cik})...")
        companyfacts = fetch_companyfacts(cik, cache_dir=cache_dir)

        for metric_name, tag_candidates in metrics:
            tag_used, series = extract_best_quarterly_series(
                companyfacts,
                tag_candidates=tag_candidates,
                min_points=args.min_points,
            )
            if len(series) < args.n_train + 1:
                print(f"[WARN] {sym} {metric_name}: not enough quarterly data ({len(series)}); skipping.")
                continue

            series = series[-(args.n_train + 1) :]  # last (train + test)
            series.sort(key=lambda p: p.end)

            y_train = np.asarray([p.val for p in series[:-1]], dtype=float)
            y_test = float(series[-1].val)
            x_train = np.arange(1, args.n_train + 1, dtype=float)
            x_test = float(args.n_train + 1)

            fits = fit_all_models(x_train, y_train)
            # Best model on training R^2
            best = max(fits, key=lambda f: f.r2_train)

            # Also collect R^2 for each type (if missing models, leave blank)
            r2_by_type = {f.model_type: f.r2_train for f in fits}
            y_pred = predict_from_fit(best, x_test)

            abs_err = float(y_pred - y_test)
            rel_err = float((y_pred - y_test) / y_test) if y_test != 0 else float("nan")

            rows.append(
                {
                    "symbol": sym,
                    "metric": metric_name,
                    "tag_used": tag_used or "",
                    "n_points_total": int(len(series)),
                    "n_train": int(args.n_train),
                    "train_ends": ",".join([p.end.strftime("%Y-%m-%d") for p in series[:-1]]),
                    "test_end": series[-1].end.strftime("%Y-%m-%d"),
                    "y_test_actual": y_test,
                    "best_model": best.model_type,
                    "r2_best_train": best.r2_train,
                    "r2_linear_train": r2_by_type.get("linear", float("nan")),
                    "r2_logarithmic_train": r2_by_type.get("logarithmic", float("nan")),
                    "r2_exponential_train": r2_by_type.get("exponential", float("nan")),
                    "r2_power_train": r2_by_type.get("power", float("nan")),
                    "y_pred_next_quarter": y_pred,
                    "error_signed": abs_err,
                    "error_abs": abs(abs_err),
                    "error_rel": rel_err,
                    "params": json.dumps(best.params, sort_keys=True),
                }
            )

    # Write CSV + JSON summary
    out_csv = out_dir / "hw4_quarterly_model_results.csv"
    fieldnames = [
        "symbol",
        "metric",
        "tag_used",
        "n_points_total",
        "n_train",
        "train_ends",
        "test_end",
        "y_test_actual",
        "best_model",
        "r2_best_train",
        "r2_linear_train",
        "r2_logarithmic_train",
        "r2_exponential_train",
        "r2_power_train",
        "y_pred_next_quarter",
        "error_signed",
        "error_abs",
        "error_rel",
        "params",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json = out_dir / "hw4_quarterly_model_results.json"
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    print(f"[INFO] Wrote results: {out_csv}")
    print(f"[INFO] Wrote results: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

