from __future__ import annotations

import numpy as np
import pandas as pd

from .state import UQState


def compute_group_gaps(
    state: UQState,
    feature_cols: list[str] | None = None,
    ref_group: str = "EU",
    target_group: str = "GS",
) -> dict[str, float]:
    cols = feature_cols or state.columns
    ref_mask = state.groups == ref_group
    target_mask = state.groups == target_group
    if not ref_mask.any() or not target_mask.any():
        raise ValueError("Both reference and target groups are required.")
    gaps: dict[str, float] = {}
    for col in cols:
        j = state.columns.index(col)
        gaps[col] = float(state.X_complete[target_mask, j].mean() - state.X_complete[ref_mask, j].mean())
    return gaps


def compute_dataframe_group_gaps(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "region",
    ref_group: str = "EU",
    target_group: str = "GS",
) -> dict[str, float]:
    ref = df[df[group_col] == ref_group]
    target = df[df[group_col] == target_group]
    if ref.empty or target.empty:
        raise ValueError("Both reference and target groups are required.")
    return {col: float(target[col].mean() - ref[col].mean()) for col in feature_cols}


def d_clean(
    state: UQState,
    clean_df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "region",
    ref_group: str = "EU",
    target_group: str = "GS",
) -> float:
    state_gaps = compute_group_gaps(state, feature_cols, ref_group, target_group)
    clean_gaps = compute_dataframe_group_gaps(clean_df, feature_cols, group_col, ref_group, target_group)
    diff = np.array([state_gaps[col] - clean_gaps[col] for col in feature_cols], dtype=float)
    return float(np.linalg.norm(diff, ord=2))


def iact(values: list[float] | np.ndarray, max_lag: int | None = None) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return 1.0
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 1e-12:
        return 1.0
    if max_lag is None:
        max_lag = min(n - 1, int(np.sqrt(n) * 10))
    tau = 1.0
    for lag in range(1, max_lag + 1):
        acov = float(np.dot(x[:-lag], x[lag:]) / (n - lag))
        rho = acov / var
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return float(max(1.0, tau))


def ess(values: list[float] | np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    n = int(np.isfinite(x).sum())
    if n == 0:
        return 0.0
    return float(n / iact(x))


def summarize_trace(logs: pd.DataFrame) -> dict[str, float]:
    if logs.empty:
        return {"acceptance_rate": float("nan")}
    summary = {"acceptance_rate": float(logs["accepted"].mean()) if "accepted" in logs else float("nan")}
    if "carbon_gap" in logs:
        summary["carbon_gap_mean"] = float(logs["carbon_gap"].mean())
        summary["carbon_gap_ess"] = ess(logs["carbon_gap"].to_numpy())
    return summary

