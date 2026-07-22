from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .proposals import DEFAULT_BOUNDS, reflect_to_bounds
from .state import UQState


def _normal_logpdf(x: float, mean: float, std: float) -> float:
    s = max(float(std), 1e-6)
    z = (float(x) - float(mean)) / s
    return -0.5 * z * z - math.log(s) - 0.5 * math.log(2.0 * math.pi)


@dataclass
class BaseImputationPosterior:
    """Group-conditional independent Gaussian imputation posterior."""

    step_scale: float = 0.35
    local_step_scale: float = 0.12
    min_group_obs: int = 3
    ref_group: str = "EU"
    target_group: str = "GS"
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BOUNDS))

    feature_cols: list[str] = field(default_factory=list)
    group_col: str = "region"
    means_: dict[tuple[str, str], float] = field(default_factory=dict)
    stds_: dict[tuple[str, str], float] = field(default_factory=dict)
    global_means_: dict[str, float] = field(default_factory=dict)
    global_stds_: dict[str, float] = field(default_factory=dict)
    observed_values_: np.ndarray | None = None
    observed_mask_: np.ndarray | None = None
    row_groups_: np.ndarray | None = None
    mean_matrix_: np.ndarray | None = None
    std_matrix_: np.ndarray | None = None

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        group_col: str,
    ) -> "BaseImputationPosterior":
        self.feature_cols = list(feature_cols)
        self.group_col = group_col
        values = df[self.feature_cols].to_numpy(dtype=float)
        self.observed_values_ = values.copy()
        self.observed_mask_ = np.isfinite(values)
        self.row_groups_ = df[group_col].to_numpy()

        for col in self.feature_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            self.global_means_[col] = float(series.mean())
            std = float(series.std(ddof=1)) if series.notna().sum() > 1 else 1.0
            self.global_stds_[col] = std if math.isfinite(std) and std > 1e-6 else 1.0

        groups = pd.Series(df[group_col]).dropna().unique().tolist()
        for group in groups:
            gdf = df[df[group_col] == group]
            for col in self.feature_cols:
                vals = pd.to_numeric(gdf[col], errors="coerce").dropna()
                if len(vals) >= self.min_group_obs:
                    mean = float(vals.mean())
                    std = float(vals.std(ddof=1)) if len(vals) > 1 else self.global_stds_[col]
                    if not math.isfinite(std) or std <= 1e-6:
                        std = self.global_stds_[col]
                else:
                    mean = self.global_means_[col]
                    std = self.global_stds_[col]
                self.means_[(str(group), col)] = mean
                self.stds_[(str(group), col)] = std

        self.mean_matrix_ = np.empty_like(values, dtype=float)
        self.std_matrix_ = np.empty_like(values, dtype=float)
        for i, group in enumerate(self.row_groups_):
            for j, col in enumerate(self.feature_cols):
                mean, std = self._mean_std(group, col)
                self.mean_matrix_[i, j] = mean
                self.std_matrix_[i, j] = max(std, 1e-6)

        return self

    def _mean_std(self, group: Any, col: str) -> tuple[float, float]:
        key = (str(group), col)
        return (
            float(self.means_.get(key, self.global_means_[col])),
            float(self.stds_.get(key, self.global_stds_[col])),
        )

    def _bounded(self, col: str, value: float) -> float:
        lower, upper = self.bounds.get(col, (-math.inf, math.inf))
        if math.isinf(lower) and math.isinf(upper):
            return float(value)
        return reflect_to_bounds(float(value), lower, upper)

    def sample_initial_state(self, df_with_missing: pd.DataFrame, rng: np.random.Generator) -> UQState:
        if not self.feature_cols:
            raise RuntimeError("BaseImputationPosterior must be fit before sampling.")
        X = df_with_missing[self.feature_cols].to_numpy(dtype=float)
        observed_mask = np.isfinite(X)
        groups = df_with_missing[self.group_col].to_numpy()
        X_complete = X.copy()

        if (
            self.mean_matrix_ is not None
            and self.std_matrix_ is not None
            and self.mean_matrix_.shape == X.shape
            and np.array_equal(groups, self.row_groups_)
        ):
            draws = rng.normal(self.mean_matrix_, self.std_matrix_)
            X_complete[~observed_mask] = draws[~observed_mask]
            for j, col in enumerate(self.feature_cols):
                lower, upper = self.bounds.get(col, (-math.inf, math.inf))
                if math.isfinite(lower) or math.isfinite(upper):
                    vals = X_complete[:, j]
                    if math.isfinite(lower) and math.isfinite(upper):
                        vals = np.clip(vals, lower, upper)
                    elif math.isfinite(lower):
                        vals = np.maximum(vals, lower)
                    else:
                        vals = np.minimum(vals, upper)
                    X_complete[:, j] = vals
        else:
            for i in range(X.shape[0]):
                for j, col in enumerate(self.feature_cols):
                    if not observed_mask[i, j]:
                        mean, std = self._mean_std(groups[i], col)
                        X_complete[i, j] = self._bounded(col, float(rng.normal(mean, std)))

        return UQState(
            X_complete=X_complete,
            observed_mask=observed_mask,
            groups=groups,
            columns=list(self.feature_cols),
        )

    def log_prob(self, state: UQState) -> float:
        missing = ~state.observed_mask
        if not missing.any():
            return 0.0

        if (
            self.mean_matrix_ is not None
            and self.std_matrix_ is not None
            and self.mean_matrix_.shape == state.X_complete.shape
            and state.columns == self.feature_cols
            and np.array_equal(state.groups, self.row_groups_)
        ):
            x = state.X_complete[missing]
            mean = self.mean_matrix_[missing]
            std = np.maximum(self.std_matrix_[missing], 1e-6)
            z = (x - mean) / std
            return float(np.sum(-0.5 * z * z - np.log(std) - 0.5 * math.log(2.0 * math.pi)))

        total = 0.0
        for i, group in enumerate(state.groups):
            for j, col in enumerate(state.columns):
                if missing[i, j]:
                    mean, std = self._mean_std(group, col)
                    total += _normal_logpdf(float(state.X_complete[i, j]), mean, std)
        return float(total)

    def _candidate_cells(self, state: UQState, proposal_type: str) -> tuple[np.ndarray, np.ndarray]:
        missing = ~state.observed_mask
        rows, cols = np.where(missing)
        if rows.size == 0:
            return rows, cols

        if proposal_type == "high_bias_cell":
            target = state.groups[rows] == self.target_group
            if target.any():
                rows = rows[target]
                cols = cols[target]
        return rows, cols

    def _cell_selection_probs(self, state: UQState, rows: np.ndarray, cols: np.ndarray, proposal_type: str) -> np.ndarray | None:
        if rows.size == 0:
            return None

        if proposal_type == "high_bias_cell":
            col_weights = np.array(
                [
                    3.0 if state.columns[c] == "carbon_kg_per_kwh" else 2.0
                    if state.columns[c] in {"repairability_score", "durability_score"}
                    else 1.0
                    for c in cols
                ],
                dtype=float,
            )
            return col_weights / col_weights.sum()

        if proposal_type == "uncertainty_cell":
            weights = np.array(
                [
                    max(self._mean_std(state.groups[int(r)], state.columns[int(c)])[1], 1e-6)
                    for r, c in zip(rows, cols)
                ],
                dtype=float,
            )
            return weights / weights.sum()

        return None

    def propose(
        self,
        state: UQState,
        proposal_type: str,
        rng: np.random.Generator,
    ) -> tuple[UQState, float, float, dict[str, Any]]:
        rows, cols = self._candidate_cells(state, proposal_type)
        if rows.size == 0:
            return state.copy(), 0.0, 0.0, {"proposal_type": proposal_type, "changed": False}

        probs = self._cell_selection_probs(state, rows, cols, proposal_type)
        if probs is not None:
            pick = int(rng.choice(np.arange(rows.size), p=probs))
        else:
            pick = int(rng.integers(rows.size))

        i = int(rows[pick])
        j = int(cols[pick])
        col = state.columns[j]
        _, std = self._mean_std(state.groups[i], col)
        scale = self.local_step_scale if proposal_type == "local" else self.step_scale
        proposed_value = float(state.X_complete[i, j] + rng.normal(0.0, max(scale * std, 1e-6)))
        proposed_value = self._bounded(col, proposed_value)

        new_state = state.copy()
        new_state.X_complete[i, j] = proposed_value
        info = {
            "proposal_type": proposal_type,
            "changed": True,
            "row": i,
            "column": col,
            "old_value": float(state.X_complete[i, j]),
            "new_value": proposed_value,
        }
        return new_state, 0.0, 0.0, info
