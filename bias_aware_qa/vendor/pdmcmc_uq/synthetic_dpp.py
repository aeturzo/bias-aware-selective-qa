from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "carbon_kg_per_kwh",
    "repairability_score",
    "durability_score",
    "recycled_content",
    "energy_efficiency",
]

GROUP_COL = "region"


def _clip_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["carbon_kg_per_kwh"] = out["carbon_kg_per_kwh"].clip(lower=1.0)
    out["repairability_score"] = out["repairability_score"].clip(0.0, 10.0)
    out["durability_score"] = out["durability_score"].clip(0.0, 10.0)
    out["recycled_content"] = out["recycled_content"].clip(0.0, 1.0)
    out["energy_efficiency"] = out["energy_efficiency"].clip(0.0, 1.0)
    return out


def generate_synthetic_dpp_uq(
    n: int = 5000,
    gs_frac: float = 0.4,
    bias_strength: float = 1.0,
    missing_eu: float = 0.10,
    missing_gs: float = 0.35,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate clean, biased, and observed-with-missingness DPP-style data."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0.0 < gs_frac < 1.0:
        raise ValueError("gs_frac must be in (0, 1).")

    rng = np.random.default_rng(seed)
    groups = np.where(rng.random(n) < gs_frac, "GS", "EU")
    is_gs = groups == "GS"

    X = np.empty((n, len(FEATURE_COLS)), dtype=float)

    eu_idx = ~is_gs
    gs_idx = is_gs

    X[eu_idx, 0] = rng.normal(70.0, 10.0, size=int(eu_idx.sum()))
    X[eu_idx, 1] = rng.normal(7.5, 1.0, size=int(eu_idx.sum()))
    X[eu_idx, 2] = rng.normal(8.0, 1.0, size=int(eu_idx.sum()))
    X[eu_idx, 3] = rng.normal(0.55, 0.12, size=int(eu_idx.sum()))
    X[eu_idx, 4] = rng.normal(0.78, 0.08, size=int(eu_idx.sum()))

    X[gs_idx, 0] = rng.normal(75.0, 12.0, size=int(gs_idx.sum()))
    X[gs_idx, 1] = rng.normal(7.2, 1.1, size=int(gs_idx.sum()))
    X[gs_idx, 2] = rng.normal(7.6, 1.1, size=int(gs_idx.sum()))
    X[gs_idx, 3] = rng.normal(0.50, 0.13, size=int(gs_idx.sum()))
    X[gs_idx, 4] = rng.normal(0.74, 0.09, size=int(gs_idx.sum()))

    clean_df = pd.DataFrame(X, columns=FEATURE_COLS)
    clean_df.insert(0, GROUP_COL, groups)
    clean_df = _clip_features(clean_df)

    biased_df = clean_df.copy()
    gs_rows = biased_df[GROUP_COL] == "GS"
    biased_df.loc[gs_rows, "carbon_kg_per_kwh"] += 20.0 * bias_strength
    biased_df.loc[gs_rows, "repairability_score"] -= 1.2 * bias_strength
    biased_df.loc[gs_rows, "durability_score"] -= 1.0 * bias_strength
    biased_df.loc[gs_rows, "recycled_content"] -= 0.10 * bias_strength
    biased_df.loc[gs_rows, "energy_efficiency"] -= 0.08 * bias_strength
    biased_df = _clip_features(biased_df)

    observed_df = biased_df.copy()
    miss_probs = np.where(groups == "GS", missing_gs, missing_eu)
    missing = rng.random((n, len(FEATURE_COLS))) < miss_probs[:, None]
    observed_mask = ~missing
    observed_df.loc[:, FEATURE_COLS] = observed_df[FEATURE_COLS].mask(missing)

    metadata = {
        "n": int(n),
        "gs_frac": float(gs_frac),
        "bias_strength": float(bias_strength),
        "missing_eu": float(missing_eu),
        "missing_gs": float(missing_gs),
        "seed": int(seed),
    }

    return {
        "clean_df": clean_df,
        "biased_df": biased_df,
        "observed_df": observed_df,
        "mask": observed_mask,
        "feature_cols": list(FEATURE_COLS),
        "group_col": GROUP_COL,
        "metadata": metadata,
    }

