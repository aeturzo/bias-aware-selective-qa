from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multi",
}

SEX_LABELS = {1: "Male", 2: "Female"}

NHANES_FEATURES = {
    "glycohemoglobin_pct": "LBXGH",
    "hdl_cholesterol_mg_dl": "LBDHDD",
    "total_cholesterol_mg_dl": "LBXTC",
    "bmi": "BMXBMI",
    "systolic_bp": "systolic_bp",
    "diastolic_bp": "diastolic_bp",
}

NHANES_BOUNDS = {
    "glycohemoglobin_pct": (3.0, 18.0),
    "hdl_cholesterol_mg_dl": (5.0, 160.0),
    "total_cholesterol_mg_dl": (50.0, 500.0),
    "bmi": (10.0, 90.0),
    "systolic_bp": (70.0, 260.0),
    "diastolic_bp": (30.0, 160.0),
}

NHANES_BIAS_DIRECTIONS = {
    "glycohemoglobin_pct": 1.0,
    "hdl_cholesterol_mg_dl": -1.0,
    "total_cholesterol_mg_dl": 1.0,
    "bmi": 1.0,
    "systolic_bp": 1.0,
    "diastolic_bp": 1.0,
}

NHANES_BIAS_MAGNITUDES = {
    "glycohemoglobin_pct": 0.60,
    "hdl_cholesterol_mg_dl": 6.0,
    "total_cholesterol_mg_dl": 15.0,
    "bmi": 2.5,
    "systolic_bp": 8.0,
    "diastolic_bp": 5.0,
}


def _read_xpt(data_dir: str | Path, name: str) -> pd.DataFrame:
    path = Path(data_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"Missing NHANES XPT file: {path}")
    return pd.read_sas(path, format="xport")


def load_nhanes_2021_2023(data_dir: str | Path) -> pd.DataFrame:
    """Load and merge the NHANES 2021-2023 lab/exam files used by this project."""

    demo = _read_xpt(data_dir, "DEMO_L.xpt")
    ghb = _read_xpt(data_dir, "GHB_L.xpt")[["SEQN", "LBXGH"]]
    hdl = _read_xpt(data_dir, "HDL_L.xpt")[["SEQN", "LBDHDD"]]
    tchol = _read_xpt(data_dir, "TCHOL_L.xpt")[["SEQN", "LBXTC"]]
    bmx = _read_xpt(data_dir, "BMX_L.xpt")[["SEQN", "BMXBMI"]]
    bpxo = _read_xpt(data_dir, "BPXO_L.xpt")
    bpxo["systolic_bp"] = bpxo[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1, skipna=True)
    bpxo["diastolic_bp"] = bpxo[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1, skipna=True)

    keep_demo = [
        "SEQN",
        "RIDRETH3",
        "RIAGENDR",
        "RIDAGEYR",
        "INDFMPIR",
        "WTMEC2YR",
        "SDMVSTRA",
        "SDMVPSU",
    ]
    df = demo[keep_demo].copy()
    for frame in [ghb, hdl, tchol, bmx, bpxo[["SEQN", "systolic_bp", "diastolic_bp"]]]:
        df = df.merge(frame, on="SEQN", how="left")

    df["race_ethnicity"] = df["RIDRETH3"].map(RACE_LABELS)
    df["sex"] = df["RIAGENDR"].map(SEX_LABELS)
    df = df.rename(columns={raw: clean for clean, raw in NHANES_FEATURES.items() if raw in df.columns})
    return df


def _task_features(task: str) -> list[str]:
    if task == "ghb_race":
        return ["glycohemoglobin_pct"]
    if task in {"cardiometabolic_race", "cardiometabolic_sex"}:
        return list(NHANES_FEATURES.keys())
    raise ValueError(
        "Unknown NHANES task. Use 'ghb_race', 'cardiometabolic_race', or 'cardiometabolic_sex'."
    )


def _task_group_defaults(task: str) -> tuple[str, str, str]:
    if task == "cardiometabolic_sex":
        return "sex", "Female", "Male"
    return "race_ethnicity", "Non-Hispanic White", "Non-Hispanic Black"


def _balanced_sample(df: pd.DataFrame, group_col: str, ref_group: str, target_group: str, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    ref = df[df[group_col] == ref_group]
    target = df[df[group_col] == target_group]
    half = max_rows // 2
    target_n = min(len(target), half)
    ref_n = min(len(ref), max_rows - target_n)
    if ref_n + target_n < max_rows:
        remaining = max_rows - ref_n - target_n
        if len(target) - target_n > len(ref) - ref_n:
            target_n = min(len(target), target_n + remaining)
        else:
            ref_n = min(len(ref), ref_n + remaining)
    pieces = [
        ref.sample(n=ref_n, random_state=int(rng.integers(1_000_000))) if ref_n else ref.head(0),
        target.sample(n=target_n, random_state=int(rng.integers(1_000_000))) if target_n else target.head(0),
    ]
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed + 17).reset_index(drop=True)


def _clip_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        lower, upper = NHANES_BOUNDS[col]
        out[col] = pd.to_numeric(out[col], errors="coerce").clip(lower=lower, upper=upper)
    return out


def prepare_nhanes_uq(
    data_dir: str | Path,
    task: str = "cardiometabolic_race",
    max_rows: int | None = 1500,
    seed: int = 0,
    ref_group: str | None = None,
    target_group: str | None = None,
    missing_ref: float = 0.10,
    missing_target: float = 0.35,
    bias_strength: float = 1.0,
) -> dict[str, Any]:
    """Build a real-data semi-synthetic NHANES UQ benchmark.

    Complete real NHANES rows are treated as clean truth. We then inject
    controlled target-group reporting/measurement bias plus group-dependent
    missingness. This keeps real lab/exam distributions while giving known
    clean values for publication-grade evaluation.
    """

    raw = load_nhanes_2021_2023(data_dir)
    feature_cols = _task_features(task)
    group_col, default_ref, default_target = _task_group_defaults(task)
    ref_group = default_ref if ref_group is None else ref_group
    target_group = default_target if target_group is None else target_group

    df = raw[raw[group_col].isin([ref_group, target_group])].copy()
    if task.startswith("ghb"):
        df = df[df["RIDAGEYR"] >= 12].copy()
    clean_df = df[[group_col] + feature_cols].dropna(subset=feature_cols).copy()
    clean_df = _clip_features(clean_df, feature_cols)
    if clean_df.empty:
        raise ValueError("No complete NHANES rows remain for the requested task/groups.")
    clean_df = _balanced_sample(clean_df, group_col, ref_group, target_group, max_rows, seed)

    biased_df = clean_df.copy()
    target_mask = biased_df[group_col] == target_group
    for col in feature_cols:
        direction = NHANES_BIAS_DIRECTIONS[col]
        magnitude = NHANES_BIAS_MAGNITUDES[col] * float(bias_strength)
        biased_df.loc[target_mask, col] = biased_df.loc[target_mask, col] + direction * magnitude
    biased_df = _clip_features(biased_df, feature_cols)

    observed_df = biased_df.copy()
    rng = np.random.default_rng(seed)
    groups = observed_df[group_col].to_numpy()
    miss_prob = np.where(groups == target_group, missing_target, missing_ref)
    missing = rng.random((len(observed_df), len(feature_cols))) < miss_prob[:, None]
    observed_df.loc[:, feature_cols] = observed_df[feature_cols].mask(missing)

    clean_gaps = {
        col: float(
            clean_df.loc[clean_df[group_col] == target_group, col].mean()
            - clean_df.loc[clean_df[group_col] == ref_group, col].mean()
        )
        for col in feature_cols
    }
    feature_stds = {col: float(clean_df[col].std(ddof=1)) for col in feature_cols}
    missingness = (
        observed_df.groupby(group_col)[feature_cols]
        .apply(lambda frame: 1.0 - frame.notna().mean())
        .reset_index()
    )

    return {
        "raw_df": raw,
        "clean_df": clean_df.reset_index(drop=True),
        "biased_df": biased_df.reset_index(drop=True),
        "observed_df": observed_df.reset_index(drop=True),
        "mask": ~missing,
        "feature_cols": feature_cols,
        "group_col": group_col,
        "ref_group": ref_group,
        "target_group": target_group,
        "target_gap": clean_gaps,
        "bounds": {col: NHANES_BOUNDS[col] for col in feature_cols},
        "metadata": {
            "task": task,
            "data_dir": str(data_dir),
            "max_rows": max_rows,
            "seed": seed,
            "missing_ref": missing_ref,
            "missing_target": missing_target,
            "bias_strength": bias_strength,
            "bias_magnitudes": {col: NHANES_BIAS_MAGNITUDES[col] for col in feature_cols},
            "bias_directions": {col: NHANES_BIAS_DIRECTIONS[col] for col in feature_cols},
            "clean_gaps": clean_gaps,
            "feature_stds": feature_stds,
            "missingness": missingness.to_dict(orient="records"),
        },
    }
