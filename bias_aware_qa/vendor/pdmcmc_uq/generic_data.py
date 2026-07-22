from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_feature_cols(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if str(c).strip()]
    return [c.strip() for c in str(raw).split(",") if c.strip()]


def parse_target_gap(raw: str | None, feature_cols: list[str]) -> dict[str, float] | None:
    if raw is None or str(raw).strip() == "":
        return None
    text = str(raw).strip()
    if Path(text).exists():
        payload = json.loads(Path(text).read_text(encoding="utf-8"))
    else:
        payload = json.loads(text)
    return {col: float(payload.get(col, 0.0)) for col in feature_cols}


def load_generic_tabular_uq(
    input_path: str | Path,
    feature_cols: list[str],
    group_col: str,
    ref_group: str,
    target_group: str,
    max_rows: int | None = None,
    clean_query: str | None = None,
    mask_rate_ref: float = 0.05,
    mask_rate_target: float = 0.20,
    seed: int = 0,
) -> dict[str, Any]:
    """Load a generic CSV/Parquet table for latent-bias UQ experiments.

    The input must contain one group column and numeric feature columns. If
    `clean_query` is provided, rows satisfying it become the clean reference;
    otherwise complete cases are used as a pragmatic evaluation reference.
    The method never downloads data and does not assume any domain-specific
    column names.
    """

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input table does not exist: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, nrows=max_rows, low_memory=False)
    else:
        raise ValueError("Use a .csv, .tsv, or .parquet input file.")

    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)

    missing = [col for col in [group_col] + feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}")

    keep = df[[group_col] + feature_cols].copy()
    keep = keep[keep[group_col].isin([ref_group, target_group])].copy()
    for col in feature_cols:
        keep[col] = pd.to_numeric(keep[col], errors="coerce")
    if keep.empty:
        raise ValueError("No rows remain after group filtering.")

    if clean_query:
        clean_df = keep.query(clean_query).dropna(subset=feature_cols).copy()
    else:
        clean_df = keep.dropna(subset=feature_cols).copy()
    if clean_df.empty:
        raise ValueError("No complete clean-reference rows available.")

    biased_df = keep.copy()
    observed_df = biased_df.copy()
    rng = np.random.default_rng(seed)
    groups = observed_df[group_col].to_numpy()
    miss_probs = np.where(groups == target_group, mask_rate_target, mask_rate_ref)
    mask_missing = rng.random((len(observed_df), len(feature_cols))) < miss_probs[:, None]
    observed_df.loc[:, feature_cols] = observed_df[feature_cols].mask(mask_missing)

    return {
        "clean_df": clean_df.reset_index(drop=True),
        "biased_df": biased_df.reset_index(drop=True),
        "observed_df": observed_df.reset_index(drop=True),
        "mask": ~mask_missing,
        "feature_cols": feature_cols,
        "group_col": group_col,
        "metadata": {
            "input_path": str(path),
            "max_rows": max_rows,
            "ref_group": ref_group,
            "target_group": target_group,
            "mask_rate_ref": mask_rate_ref,
            "mask_rate_target": mask_rate_target,
            "seed": seed,
        },
    }

