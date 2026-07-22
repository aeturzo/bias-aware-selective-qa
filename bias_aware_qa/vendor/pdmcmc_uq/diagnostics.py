from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import ess, iact


try:  # Optional; the internal rank-normalized fallback is used otherwise.
    import arviz as az
except Exception:  # pragma: no cover - depends on optional local install
    az = None


def autocorrelation(values: np.ndarray, max_lag: int = 80) -> pd.DataFrame:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return pd.DataFrame({"lag": [0], "autocorrelation": [1.0]})
    x = x - x.mean()
    var = float(np.dot(x, x) / x.size)
    rows = [{"lag": 0, "autocorrelation": 1.0}]
    if var <= 1e-12:
        rows.extend({"lag": lag, "autocorrelation": 0.0} for lag in range(1, max_lag + 1))
        return pd.DataFrame(rows)
    for lag in range(1, min(max_lag, x.size - 1) + 1):
        acov = float(np.dot(x[:-lag], x[lag:]) / (x.size - lag))
        rows.append({"lag": lag, "autocorrelation": acov / var})
    return pd.DataFrame(rows)


def _split_chains(chains: list[np.ndarray]) -> list[np.ndarray]:
    split: list[np.ndarray] = []
    for chain in chains:
        x = np.asarray(chain, dtype=float)
        x = x[np.isfinite(x)]
        half = x.size // 2
        if half >= 2:
            split.append(x[:half])
            split.append(x[-half:])
    return split


def rhat(chains: list[np.ndarray]) -> float:
    split = _split_chains(chains)
    if len(split) < 2:
        return float("nan")
    n = min(len(x) for x in split)
    if n < 2:
        return float("nan")
    matrix = np.vstack([x[:n] for x in split])
    chain_means = matrix.mean(axis=1)
    chain_vars = matrix.var(axis=1, ddof=1)
    w = float(chain_vars.mean())
    if w <= 1e-12:
        return 1.0
    b = float(n * chain_means.var(ddof=1))
    var_hat = ((n - 1.0) / n) * w + b / n
    return float(np.sqrt(max(var_hat / w, 0.0)))


def _normal_scores(u: np.ndarray) -> np.ndarray:
    clipped = np.clip(u, 1e-9, 1.0 - 1e-9)
    try:
        from scipy.stats import norm

        return np.asarray(norm.ppf(clipped), dtype=float)
    except Exception:  # pragma: no cover - only used when scipy is unavailable
        return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=float)


def _rank_normalize(chains: list[np.ndarray]) -> list[np.ndarray]:
    finite = [np.asarray(x, dtype=float)[np.isfinite(x)] for x in chains]
    lengths = [len(x) for x in finite]
    if sum(lengths) < 4:
        return finite
    pooled = np.concatenate(finite)
    ranks = pd.Series(pooled).rank(method="average").to_numpy(dtype=float)
    u = (ranks - 0.375) / (len(pooled) + 0.25)
    z = _normal_scores(u)
    out: list[np.ndarray] = []
    start = 0
    for length in lengths:
        out.append(z[start : start + length])
        start += length
    return out


def rank_normalized_rhat(chains: list[np.ndarray]) -> float:
    """Approximate Vehtari-style rank-normalized split R-hat.

    If ArviZ is installed, `diagnostics_table` uses ArviZ directly. This
    fallback rank-normalizes draws and returns the maximum of the ordinary and
    folded split R-hat values, matching the practical diagnostic intent.
    """

    ranked = _rank_normalize(chains)
    if len(ranked) < 2:
        return float("nan")
    base = rhat(ranked)
    pooled = np.concatenate([x for x in ranked if len(x)])
    if pooled.size < 4:
        return base
    median = float(np.median(pooled))
    folded = [np.abs(x - median) for x in ranked]
    folded_rhat = rhat(folded)
    return float(np.nanmax([base, folded_rhat]))


def multi_chain_ess(chains: list[np.ndarray]) -> float:
    finite = [np.asarray(x, dtype=float)[np.isfinite(x)] for x in chains]
    finite = [x for x in finite if x.size > 2]
    if not finite:
        return 0.0
    total_n = sum(len(x) for x in finite)
    tau = float(np.mean([iact(x) for x in finite]))
    return float(total_n / max(tau, 1.0))


def tail_ess(chains: list[np.ndarray]) -> float:
    finite = [np.asarray(x, dtype=float)[np.isfinite(x)] for x in chains]
    finite = [x for x in finite if x.size > 2]
    if not finite:
        return 0.0
    pooled = np.concatenate(finite)
    lo, hi = np.quantile(pooled, [0.05, 0.95])
    lower_ind = [(x <= lo).astype(float) for x in finite]
    upper_ind = [(x >= hi).astype(float) for x in finite]
    return float(min(multi_chain_ess(lower_ind), multi_chain_ess(upper_ind)))


def _arviz_diagnostics(chains: list[np.ndarray]) -> tuple[float, float, float] | None:
    if az is None:
        return None
    finite = [np.asarray(x, dtype=float)[np.isfinite(x)] for x in chains if np.isfinite(x).sum() > 2]
    if len(finite) < 2:
        return None
    n = min(len(x) for x in finite)
    if n < 4:
        return None
    arr = np.vstack([x[:n] for x in finite])
    try:
        rhat_value = float(az.rhat(arr, method="rank").item())
        ess_bulk_value = float(az.ess(arr, method="bulk").item())
        ess_tail_value = float(az.ess(arr, method="tail").item())
    except Exception:
        return None
    return rhat_value, ess_bulk_value, ess_tail_value


def diagnostics_table(
    logs: pd.DataFrame,
    quantity_cols: list[str],
    method_col: str = "method",
    chain_col: str = "chain",
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    sample = logs[logs["phase"] == "sample"].copy() if "phase" in logs else logs.copy()
    for method, method_df in sample.groupby(method_col):
        runtime = float(method_df["runtime_seconds"].max()) if "runtime_seconds" in method_df else float("nan")
        for col in quantity_cols:
            if col not in method_df:
                continue
            chains = [
                chain_df[col].to_numpy(dtype=float)
                for _, chain_df in method_df.groupby(chain_col)
                if chain_df[col].notna().sum() > 2
            ]
            pooled = method_df[col].to_numpy(dtype=float)
            split_rhat = rhat(chains)
            rank_rhat = rank_normalized_rhat(chains)
            custom_bulk = multi_chain_ess(chains)
            custom_tail = tail_ess(chains)
            arviz_values = _arviz_diagnostics(chains)
            if arviz_values is not None:
                diagnostic_backend = "arviz_rank"
                final_rhat, bulk, tail = arviz_values
            else:
                diagnostic_backend = "internal_rank_fallback"
                final_rhat, bulk, tail = rank_rhat, custom_bulk, custom_tail
            rows.append(
                {
                    "method": method,
                    "quantity": col,
                    "diagnostic_backend": diagnostic_backend,
                    "rhat": final_rhat,
                    "rhat_split": split_rhat,
                    "rhat_rank_fallback": rank_rhat,
                    "ess_bulk": bulk,
                    "ess_bulk_custom": custom_bulk,
                    "ess_tail": tail,
                    "ess_tail_custom": custom_tail,
                    "iact_mean": float(np.mean([iact(x) for x in chains])) if chains else float("nan"),
                    "single_trace_ess": ess(pooled),
                    "acceptance_rate": float(method_df["accepted"].mean()) if "accepted" in method_df else float("nan"),
                    "runtime_seconds": runtime,
                    "ess_bulk_per_second": bulk / runtime if runtime and runtime > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)
