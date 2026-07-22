"""Post-hoc confidence calibration (isotonic / pool-adjacent-violators).

The fused confidence is monotone but *under-confident* (the system is right more
often than its raw score claims), which inflates ECE. We fit a monotone map from
raw confidence to empirical correctness on a DEV split and apply it to the
held-out TEST split, then report calibrated ECE. This is standard post-hoc
calibration; no external dependency (PAV implemented in numpy).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import selective


def _pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators isotonic regression of y on sorted x.

    Returns (x_sorted, fitted_y) giving a non-decreasing step function.
    """
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order].astype(float), y[order].astype(float)
    n = ys.size
    # weights and level values for the PAV blocks
    val = ys.copy()
    wt = np.ones(n, dtype=float)
    idx = list(range(n))  # block boundaries as a simple stack
    i = 0
    blocks: list[list[float]] = []  # [value, weight, count]
    for k in range(n):
        cur = [val[k], 1.0, 1.0]
        while blocks and blocks[-1][0] >= cur[0]:
            prev = blocks.pop()
            tot_w = prev[1] + cur[1]
            cur = [(prev[0] * prev[1] + cur[0] * cur[1]) / tot_w, tot_w, prev[2] + cur[2]]
        blocks.append(cur)
    fitted = np.empty(n, dtype=float)
    pos = 0
    for value, _w, count in blocks:
        c = int(count)
        fitted[pos:pos + c] = value
        pos += c
    return xs, fitted


class IsotonicCalibrator:
    """Monotone confidence -> P(correct) calibrator."""

    def __init__(self) -> None:
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, confidence: np.ndarray, correct: np.ndarray) -> "IsotonicCalibrator":
        x = np.asarray(confidence, dtype=float)
        y = np.asarray(correct, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size == 0:
            self._x, self._y = np.array([0.0, 1.0]), np.array([0.0, 1.0])
            return self
        self._x, self._y = _pav(x, y)
        return self

    def predict(self, confidence: np.ndarray) -> np.ndarray:
        x = np.asarray(confidence, dtype=float)
        if self._x is None:
            return x
        return np.interp(x, self._x, self._y, left=self._y[0], right=self._y[-1])


def calibrated_report(
    rows: list[dict[str, Any]], dev_seeds: list[int], test_seeds: list[int],
) -> dict[str, Any]:
    """Fit calibration on dev seeds, evaluate ECE on test seeds.

    Uses only answered rows with a finite (confidence, correct). Returns raw vs
    calibrated ECE on the held-out test split and the calibrator's knots.
    """
    ans = [r for r in rows if r.get("answered") and np.isfinite(r.get("confidence", np.nan))]
    dev = [r for r in ans if r.get("seed") in set(dev_seeds)]
    test = [r for r in ans if r.get("seed") in set(test_seeds)]
    if not dev or not test:
        # fall back to a random 60/40 split if seed split is empty
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(ans))
        cut = int(0.6 * len(ans))
        dev = [ans[i] for i in idx[:cut]]
        test = [ans[i] for i in idx[cut:]]

    cal = IsotonicCalibrator().fit(
        np.array([r["confidence"] for r in dev]),
        np.array([bool(r["correct"]) for r in dev]),
    )
    test_conf = np.array([r["confidence"] for r in test])
    test_correct = np.array([bool(r["correct"]) for r in test])
    test_conf_cal = cal.predict(test_conf)
    return {
        "n_dev": len(dev), "n_test": len(test),
        "ece_raw_test": round(selective.ece(test_conf, test_correct), 4),
        "ece_calibrated_test": round(selective.ece(test_conf_cal, test_correct), 4),
        "mean_conf_raw_test": round(float(test_conf.mean()), 4),
        "mean_conf_cal_test": round(float(test_conf_cal.mean()), 4),
        "test_accuracy": round(float(test_correct.mean()), 4),
    }


def brier(confidence: np.ndarray, correct: np.ndarray) -> float:
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=float)
    mask = np.isfinite(c) & np.isfinite(y)
    if not mask.any():
        return float("nan")
    return float(np.mean((c[mask] - y[mask]) ** 2))


def loso_calibration_report(rows: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """Leave-one-seed-out calibration: for each seed, fit isotonic on the other
    seeds and evaluate on the held-out seed. Reports mean +/- sd of held-out
    calibrated ECE and Brier score -- a far less fragile summary than a single
    tiny test split.
    """
    ans = [r for r in rows if r.get("answered") and np.isfinite(r.get("confidence", np.nan))]
    per_seed = []
    for s in seeds:
        dev = [r for r in ans if r.get("seed") != s]
        test = [r for r in ans if r.get("seed") == s]
        if not dev or not test:
            continue
        cal = IsotonicCalibrator().fit(
            np.array([r["confidence"] for r in dev]),
            np.array([bool(r["correct"]) for r in dev]))
        conf = np.array([r["confidence"] for r in test])
        corr = np.array([bool(r["correct"]) for r in test])
        conf_cal = cal.predict(conf)
        per_seed.append({
            "seed": s, "n_test": len(test),
            "ece_raw": selective.ece(conf, corr),
            "ece_cal": selective.ece(conf_cal, corr),
            "brier_raw": brier(conf, corr),
            "brier_cal": brier(conf_cal, corr),
            "acc": float(corr.mean()),
        })
    if not per_seed:
        return {"folds": [], "n_folds": 0}
    agg = {}
    for key in ("ece_raw", "ece_cal", "brier_raw", "brier_cal", "acc"):
        vals = np.array([f[key] for f in per_seed], dtype=float)
        agg[f"{key}_mean"] = round(float(np.nanmean(vals)), 4)
        agg[f"{key}_sd"] = round(float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0, 4)
    return {"folds": per_seed, "n_folds": len(per_seed), **agg}
