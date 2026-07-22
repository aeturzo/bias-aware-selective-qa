from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .base_posterior import BaseImputationPosterior
from .constraints import Constraint
from .metrics import compute_dataframe_group_gaps, compute_group_gaps, d_clean
from .sampler import ConstraintCalibratedPDMCMCUQ, SamplerConfig
from .state import UQState


@dataclass
class MethodResult:
    method: str
    logs: pd.DataFrame
    lambda_trajectory: pd.DataFrame = field(default_factory=pd.DataFrame)
    agent_logs: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_lambdas: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0


def state_quantities(
    state: UQState,
    feature_cols: list[str],
    clean_df: pd.DataFrame | None = None,
    group_col: str = "region",
) -> dict[str, float]:
    gaps = compute_group_gaps(state, feature_cols)
    values = {
        "gap_carbon_kg_per_kwh": gaps.get("carbon_kg_per_kwh", float("nan")),
        "gap_repairability_score": gaps.get("repairability_score", float("nan")),
        "gap_durability_score": gaps.get("durability_score", float("nan")),
        "mean_carbon": float(state.X_complete[:, state.columns.index("carbon_kg_per_kwh")].mean())
        if "carbon_kg_per_kwh" in state.columns
        else float("nan"),
    }
    if clean_df is not None:
        values["d_clean"] = d_clean(state, clean_df, feature_cols, group_col=group_col)
    return values


def _sample_p0_logs(
    method: str,
    posterior: BaseImputationPosterior,
    observed_df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    n_samples: int,
    seed: int,
    clean_df: pd.DataFrame | None = None,
) -> MethodResult:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for iteration in range(int(n_samples)):
        state = posterior.sample_initial_state(observed_df, rng)
        rows.append(
            {
                "iteration": iteration,
                "phase": "sample",
                "accepted": True,
                "acceptance_prob": 1.0,
                **state_quantities(state, feature_cols, clean_df=clean_df, group_col=group_col),
            }
        )
    return MethodResult(method=method, logs=pd.DataFrame(rows), runtime_seconds=time.perf_counter() - start)


def naive_p0(
    posterior: BaseImputationPosterior,
    observed_df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    n_samples: int,
    seed: int,
    clean_df: pd.DataFrame | None = None,
) -> MethodResult:
    return _sample_p0_logs("naive_p0", posterior, observed_df, feature_cols, group_col, n_samples, seed, clean_df)


def multiple_imputation(
    posterior: BaseImputationPosterior,
    observed_df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    n_samples: int,
    seed: int,
    clean_df: pd.DataFrame | None = None,
) -> MethodResult:
    return _sample_p0_logs(
        "multiple_imputation",
        posterior,
        observed_df,
        feature_cols,
        group_col,
        n_samples,
        seed + 701,
        clean_df,
    )


def bootstrap(
    posterior: BaseImputationPosterior,
    observed_df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    n_samples: int,
    seed: int,
    clean_df: pd.DataFrame | None = None,
) -> MethodResult:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    base_state = posterior.sample_initial_state(observed_df, rng)
    n = base_state.X_complete.shape[0]
    for iteration in range(int(n_samples)):
        idx = rng.integers(0, n, size=n)
        boot_state = UQState(
            X_complete=base_state.X_complete[idx].copy(),
            observed_mask=np.ones_like(base_state.X_complete[idx], dtype=bool),
            groups=base_state.groups[idx].copy(),
            columns=list(base_state.columns),
        )
        rows.append(
            {
                "iteration": iteration,
                "phase": "sample",
                "accepted": True,
                "acceptance_prob": 1.0,
                **state_quantities(boot_state, feature_cols, clean_df=clean_df, group_col=group_col),
            }
        )
    return MethodResult(method="bootstrap", logs=pd.DataFrame(rows), runtime_seconds=time.perf_counter() - start)


def run_pdmcmc_method(
    method: str,
    posterior: BaseImputationPosterior,
    constraints_factory: Callable[[], list[Constraint]],
    observed_df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    config: SamplerConfig,
    seed: int,
    clean_df: pd.DataFrame | None = None,
    initial_lambdas: dict[str, float] | None = None,
) -> MethodResult:
    rng = np.random.default_rng(seed + 1234)
    initial_state = posterior.sample_initial_state(observed_df, rng)
    constraints = constraints_factory()
    if initial_lambdas:
        for c in constraints:
            if c.name in initial_lambdas:
                c.lambda_value = float(initial_lambdas[c.name])
    config.method = method
    config.seed = seed
    sampler = ConstraintCalibratedPDMCMCUQ(posterior, constraints, config)
    start = time.perf_counter()
    output = sampler.run(initial_state)
    runtime = time.perf_counter() - start
    logs = output["final_result"].logs.copy()
    if clean_df is not None and not logs.empty:
        clean_gaps = compute_dataframe_group_gaps(clean_df, feature_cols, group_col=group_col)
        sample_mask = logs["phase"] == "sample"
        if sample_mask.any():
            diff_sq = np.zeros(int(sample_mask.sum()), dtype=float)
            for col in feature_cols:
                log_col = f"gap_{col}"
                if log_col in logs.columns:
                    diff = logs.loc[sample_mask, log_col].to_numpy(dtype=float) - float(clean_gaps[col])
                    diff_sq += diff * diff
            logs.loc[sample_mask, "d_clean"] = np.sqrt(diff_sq)
    return MethodResult(
        method=method,
        logs=logs,
        lambda_trajectory=output["lambda_trajectory"],
        agent_logs=output["agent_logs"],
        final_lambdas=output["final_lambdas"],
        runtime_seconds=runtime,
    )
