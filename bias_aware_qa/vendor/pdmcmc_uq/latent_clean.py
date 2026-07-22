from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .proposals import DEFAULT_BOUNDS, reflect_to_bounds
from .state import UQState


def _normal_logpdf_array(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    s = np.maximum(std, 1e-6)
    z = (x - mean) / s
    return float(np.sum(-0.5 * z * z - np.log(s) - 0.5 * math.log(2.0 * math.pi)))


@dataclass
class LatentCleanBiasPosterior:
    """Latent-clean posterior with explicit target-group measurement bias.

    Observation model for observed cells:

        y_ij = x_clean_ij + sign_j * b_j * 1[g_i == target] + noise_ij

    where b_j >= 0 and sign_j encodes the known bias direction. The latent
    clean table can move even for observed cells; observations are enforced
    probabilistically through the likelihood instead of being fixed.
    """

    step_scale: float = 0.18
    local_step_scale: float = 0.06
    bias_step_scale: float = 0.08
    obs_noise_scale: float = 0.20
    high_bias_param_prob: float = 0.20
    uncertainty_param_prob: float = 0.30
    local_param_prob: float = 0.05
    block_bias_param_prob: float = 0.0
    ref_group: str = "EU"
    target_group: str = "GS"
    target_gap: dict[str, float] | None = None
    use_target_gap_in_prior: bool = False
    bounds: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_BOUNDS))

    feature_cols: list[str] = field(default_factory=list)
    group_col: str = "region"
    observed_values_: np.ndarray | None = None
    observed_mask_: np.ndarray | None = None
    row_groups_: np.ndarray | None = None
    prior_mean_: np.ndarray | None = None
    prior_std_: np.ndarray | None = None
    obs_std_: np.ndarray | None = None
    initial_bias_: dict[str, float] = field(default_factory=dict)
    bias_prior_scale_: dict[str, float] = field(default_factory=dict)

    def _sign(self, col: str) -> float:
        lower = col.lower()
        adverse_high_tokens = (
            "carbon",
            "footprint",
            "emission",
            "salt",
            "sugar",
            "glyco",
            "glucose",
            "a1c",
            "total_cholesterol",
            "cholesterol_total",
            "bmi",
            "systolic",
            "diastolic",
            "blood_pressure",
        )
        protective_high_tokens = ("hdl",)
        if any(token in lower for token in protective_high_tokens):
            return -1.0
        if any(token in lower for token in adverse_high_tokens):
            return 1.0
        return -1.0

    def _target_gap(self, col: str) -> float:
        if self.target_gap is None:
            return 0.0
        return float(self.target_gap.get(col, 0.0))

    def fit(self, df: pd.DataFrame, feature_cols: list[str], group_col: str) -> "LatentCleanBiasPosterior":
        self.feature_cols = list(feature_cols)
        self.group_col = group_col
        values = df[self.feature_cols].to_numpy(dtype=float)
        self.observed_values_ = values.copy()
        self.observed_mask_ = np.isfinite(values)
        self.row_groups_ = df[group_col].to_numpy()

        n, p = values.shape
        self.prior_mean_ = np.empty((n, p), dtype=float)
        self.prior_std_ = np.empty((n, p), dtype=float)
        self.obs_std_ = np.empty((n, p), dtype=float)

        ref_df = df[df[group_col] == self.ref_group]
        tgt_df = df[df[group_col] == self.target_group]
        if ref_df.empty or tgt_df.empty:
            raise ValueError("LatentCleanBiasPosterior requires both reference and target groups.")

        for j, col in enumerate(self.feature_cols):
            ref_vals = pd.to_numeric(ref_df[col], errors="coerce").dropna()
            tgt_vals = pd.to_numeric(tgt_df[col], errors="coerce").dropna()
            all_vals = pd.to_numeric(df[col], errors="coerce").dropna()
            ref_mean = float(ref_vals.mean()) if len(ref_vals) else float(all_vals.mean())
            ref_std = float(ref_vals.std(ddof=1)) if len(ref_vals) > 1 else float(all_vals.std(ddof=1))
            tgt_mean_obs = float(tgt_vals.mean()) if len(tgt_vals) else ref_mean
            if not math.isfinite(ref_std) or ref_std <= 1e-6:
                ref_std = float(all_vals.std(ddof=1)) if len(all_vals) > 1 else 1.0
            if not math.isfinite(ref_std) or ref_std <= 1e-6:
                ref_std = 1.0

            sign = self._sign(col)
            policy_target_mean = ref_mean + self._target_gap(col)
            unconstrained_target_mean = tgt_mean_obs
            clean_target_mean = (
                policy_target_mean if self.use_target_gap_in_prior else unconstrained_target_mean
            )
            policy_implied_bias = max(0.0, sign * (tgt_mean_obs - policy_target_mean))
            init_bias = policy_implied_bias if self.use_target_gap_in_prior else 0.0
            self.initial_bias_[col] = float(init_bias)
            self.bias_prior_scale_[col] = max(abs(policy_implied_bias) * 2.0, ref_std * 2.0, 1e-3)

            ref_mask = self.row_groups_ == self.ref_group
            tgt_mask = self.row_groups_ == self.target_group
            self.prior_mean_[ref_mask, j] = ref_mean
            self.prior_mean_[tgt_mask, j] = clean_target_mean
            self.prior_std_[:, j] = max(ref_std * 1.5, 1e-3)
            self.obs_std_[:, j] = max(ref_std * self.obs_noise_scale, 1e-3)

        return self

    def _bounded(self, col: str, value: float) -> float:
        lower, upper = self.bounds.get(col, (-math.inf, math.inf))
        if math.isinf(lower) and math.isinf(upper):
            return float(value)
        return reflect_to_bounds(float(value), lower, upper)

    def _clip_column(self, col: str, values: np.ndarray) -> np.ndarray:
        lower, upper = self.bounds.get(col, (-math.inf, math.inf))
        out = values
        if math.isfinite(lower):
            out = np.maximum(out, lower)
        if math.isfinite(upper):
            out = np.minimum(out, upper)
        return out

    def sample_initial_state(self, df_with_missing: pd.DataFrame, rng: np.random.Generator) -> UQState:
        if self.prior_mean_ is None or self.prior_std_ is None:
            raise RuntimeError("LatentCleanBiasPosterior must be fit before sampling.")
        observed = df_with_missing[self.feature_cols].to_numpy(dtype=float)
        observed_mask = np.isfinite(observed)
        groups = df_with_missing[self.group_col].to_numpy()
        X = rng.normal(self.prior_mean_, self.prior_std_)

        bias_params = dict(self.initial_bias_)
        target_mask = groups == self.target_group
        for j, col in enumerate(self.feature_cols):
            sign = self._sign(col)
            obs_col = observed[:, j]
            use_obs = observed_mask[:, j]
            X[use_obs, j] = obs_col[use_obs]
            target_obs = use_obs & target_mask
            X[target_obs, j] = obs_col[target_obs] - sign * bias_params[col]
            X[:, j] = self._clip_column(col, X[:, j])

        return UQState(
            X_complete=X,
            observed_mask=observed_mask,
            groups=groups,
            columns=list(self.feature_cols),
            bias_params=bias_params,
        )

    def _measurement_mean(self, state: UQState) -> np.ndarray:
        if state.bias_params is None:
            raise ValueError("Latent clean state must contain bias_params.")
        mean = state.X_complete.copy()
        target_mask = state.groups == self.target_group
        for j, col in enumerate(state.columns):
            mean[target_mask, j] += self._sign(col) * float(state.bias_params.get(col, 0.0))
        return mean

    def log_prob(self, state: UQState) -> float:
        if (
            self.prior_mean_ is None
            or self.prior_std_ is None
            or self.obs_std_ is None
            or self.observed_values_ is None
            or self.observed_mask_ is None
        ):
            raise RuntimeError("LatentCleanBiasPosterior must be fit before log_prob.")

        lp = _normal_logpdf_array(state.X_complete, self.prior_mean_, self.prior_std_)
        obs_mean = self._measurement_mean(state)
        mask = self.observed_mask_
        lp += _normal_logpdf_array(self.observed_values_[mask], obs_mean[mask], self.obs_std_[mask])

        if state.bias_params is None:
            return float("-inf")
        for col in self.feature_cols:
            b = float(state.bias_params.get(col, 0.0))
            if b < 0.0:
                return float("-inf")
            scale = max(float(self.bias_prior_scale_.get(col, 1.0)), 1e-6)
            lp += -0.5 * (b / scale) ** 2 - math.log(scale) - 0.5 * math.log(2.0 * math.pi) + math.log(2.0)
        return float(lp)

    def _candidate_cells(self, state: UQState, proposal_type: str) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = np.where(np.ones_like(state.X_complete, dtype=bool))
        if proposal_type == "high_bias_cell":
            target = state.groups[rows] == self.target_group
            if target.any():
                rows = rows[target]
                cols = cols[target]
        elif proposal_type == "uncertainty_cell":
            missing = ~state.observed_mask
            m_rows, m_cols = np.where(missing)
            if m_rows.size:
                rows, cols = m_rows, m_cols
        return rows, cols

    def _propose_bias_param(self, state: UQState, rng: np.random.Generator) -> tuple[UQState, dict[str, Any]]:
        if state.bias_params is None:
            raise ValueError("Latent clean state must contain bias_params.")
        col = str(rng.choice(state.columns))
        scale = max(self.bias_step_scale * self.bias_prior_scale_.get(col, 1.0), 1e-5)
        old = float(state.bias_params.get(col, 0.0))
        new = abs(old + float(rng.normal(0.0, scale)))
        proposed = state.copy()
        proposed.bias_params = dict(state.bias_params)
        proposed.bias_params[col] = new
        j = state.columns.index(col)
        target_mask = state.groups == self.target_group
        delta = new - old
        proposed.X_complete[target_mask, j] = self._clip_column(
            col,
            proposed.X_complete[target_mask, j] - self._sign(col) * delta,
        )
        return proposed, {
            "proposal_type": "bias_param",
            "changed": True,
            "column": col,
            "old_value": old,
            "new_value": new,
        }

    def _propose_bias_block(self, state: UQState, rng: np.random.Generator) -> tuple[UQState, dict[str, Any]]:
        if state.bias_params is None:
            raise ValueError("Latent clean state must contain bias_params.")
        proposed = state.copy()
        proposed.bias_params = dict(state.bias_params)
        target_mask = state.groups == self.target_group
        changed: list[str] = []
        for col in state.columns:
            scale = max(self.bias_step_scale * self.bias_prior_scale_.get(col, 1.0), 1e-5)
            old = float(state.bias_params.get(col, 0.0))
            new = abs(old + float(rng.normal(0.0, scale)))
            proposed.bias_params[col] = new
            j = state.columns.index(col)
            delta = new - old
            proposed.X_complete[target_mask, j] = self._clip_column(
                col,
                proposed.X_complete[target_mask, j] - self._sign(col) * delta,
            )
            changed.append(col)
        return proposed, {
            "proposal_type": "bias_block",
            "changed": True,
            "column": ",".join(changed),
        }

    def propose(
        self,
        state: UQState,
        proposal_type: str,
        rng: np.random.Generator,
    ) -> tuple[UQState, float, float, dict[str, Any]]:
        if (
            proposal_type == "bias_param"
            or (proposal_type == "high_bias_cell" and rng.random() < self.high_bias_param_prob)
            or (proposal_type == "uncertainty_cell" and rng.random() < self.uncertainty_param_prob)
            or (proposal_type == "local" and rng.random() < self.local_param_prob)
        ):
            if rng.random() < self.block_bias_param_prob and len(state.columns) > 1:
                proposed, info = self._propose_bias_block(state, rng)
            else:
                proposed, info = self._propose_bias_param(state, rng)
            return proposed, 0.0, 0.0, info

        rows, cols = self._candidate_cells(state, proposal_type)
        if rows.size == 0:
            return state.copy(), 0.0, 0.0, {"proposal_type": proposal_type, "changed": False}

        if proposal_type == "high_bias_cell":
            weights = np.array(
                [
                    3.0 if state.columns[c] == "carbon_kg_per_kwh" else 2.0
                    if state.columns[c] in {"repairability_score", "durability_score"}
                    else 1.0
                    for c in cols
                ],
                dtype=float,
            )
            pick = int(rng.choice(np.arange(rows.size), p=weights / weights.sum()))
        else:
            pick = int(rng.integers(rows.size))

        i = int(rows[pick])
        j = int(cols[pick])
        col = state.columns[j]
        std = float(self.prior_std_[i, j]) if self.prior_std_ is not None else 1.0
        scale = self.local_step_scale if proposal_type == "local" else self.step_scale
        proposed_value = float(state.X_complete[i, j] + rng.normal(0.0, max(scale * std, 1e-6)))
        proposed_value = self._bounded(col, proposed_value)
        proposed = state.copy()
        proposed.X_complete[i, j] = proposed_value
        return proposed, 0.0, 0.0, {
            "proposal_type": proposal_type,
            "changed": True,
            "row": i,
            "column": col,
            "old_value": float(state.X_complete[i, j]),
            "new_value": proposed_value,
        }
