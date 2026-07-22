from __future__ import annotations

from statistics import NormalDist

import numpy as np


def credible_interval(values: list[float] | np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return (
        float(np.quantile(x, alpha / 2.0)),
        float(np.quantile(x, 1.0 - alpha / 2.0)),
    )


def interval_summary(
    values: list[float] | np.ndarray,
    clean_value: float,
    alpha: float = 0.05,
    extra_sd: float = 0.0,
) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "posterior_mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "ci_width": float("nan"),
            "stat_ci_lower": float("nan"),
            "stat_ci_upper": float("nan"),
            "stat_ci_width": float("nan"),
            "bias_uncertainty_sd": float(extra_sd),
            "total_sd": float("nan"),
            "clean_value": float(clean_value),
            "center_error_to_clean": float("nan"),
            "ci_covers_clean": 0.0,
        }
    stat_lo, stat_hi = credible_interval(x, alpha=alpha)
    mean = float(np.mean(x))
    if extra_sd > 0.0:
        stat_var = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
        total_sd = float(np.sqrt(max(stat_var, 0.0) + float(extra_sd) ** 2))
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        lo = float(mean - z * total_sd)
        hi = float(mean + z * total_sd)
    else:
        total_sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        lo, hi = stat_lo, stat_hi
    cover_tol = 1e-12 * max(1.0, abs(float(clean_value)), abs(lo), abs(hi))
    return {
        "posterior_mean": mean,
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_width": float(hi - lo),
        "stat_ci_lower": stat_lo,
        "stat_ci_upper": stat_hi,
        "stat_ci_width": float(stat_hi - stat_lo),
        "bias_uncertainty_sd": float(extra_sd),
        "total_sd": total_sd,
        "clean_value": float(clean_value),
        "center_error_to_clean": float(abs(mean - clean_value)),
        "ci_covers_clean": float((lo - cover_tol) <= clean_value <= (hi + cover_tol)),
    }
