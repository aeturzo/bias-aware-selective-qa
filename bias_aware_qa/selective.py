"""Self-contained selective-prediction and calibration metrics.

Definitions match the COMPASS evaluation harness:
  * AURC  = area under the risk-coverage curve (trapezoid of risk vs coverage),
            where risk = 1 - accuracy among answered items.
  * ECE   = quantile-binned expected calibration error (avoids empty bins).
These are re-implemented here (a few dozen lines each) so the package does not
depend on the COMPASS server code.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def risk_coverage_curve(
    confidence: Sequence[float], correct: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coverage, risk) arrays by sweeping a confidence threshold.

    Items are ranked by confidence (high first); at each prefix we report the
    coverage (fraction answered) and the risk (error rate among answered).
    """
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=bool)
    n = conf.size
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    order = np.argsort(-conf)
    corr_sorted = corr[order]
    coverages, risks = [], []
    cum_correct = 0
    for k in range(1, n + 1):
        cum_correct += int(corr_sorted[k - 1])
        coverages.append(k / n)
        risks.append(1.0 - cum_correct / k)
    return np.asarray(coverages), np.asarray(risks)


def aurc(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    cov, risk = risk_coverage_curve(confidence, correct)
    if cov.size < 2:
        return float("nan")
    return float(np.trapz(risk, cov))


def ece(
    confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Quantile-binned expected calibration error (lower is better)."""
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    n = conf.size
    if n == 0:
        return float("nan")
    order = np.argsort(conf)
    conf, corr = conf[order], corr[order]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        sb_conf = conf[lo:hi]
        sb_corr = corr[lo:hi]
        acc = float(sb_corr.mean())
        cf = float(sb_conf.mean())
        total += abs(acc - cf) * (len(sb_conf) / n)
    return float(total)


def reliability_points(
    confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """Return (mean_confidence, accuracy, count) per quantile bin for plotting."""
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    n = conf.size
    if n == 0:
        return []
    order = np.argsort(conf)
    conf, corr = conf[order], corr[order]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    pts = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        pts.append((float(conf[lo:hi].mean()), float(corr[lo:hi].mean()), int(hi - lo)))
    return pts


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (float(center - half), float(center + half))
