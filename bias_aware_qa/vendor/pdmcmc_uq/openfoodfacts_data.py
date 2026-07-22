from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EU_MARKERS = {
    "en:france",
    "en:germany",
    "en:spain",
    "en:italy",
    "en:netherlands",
    "en:belgium",
    "en:sweden",
    "en:denmark",
    "en:finland",
    "en:poland",
    "en:austria",
    "en:ireland",
    "en:portugal",
    "en:greece",
    "en:czech-republic",
    "en:slovakia",
    "en:slovenia",
    "en:croatia",
    "en:hungary",
    "en:romania",
    "en:bulgaria",
    "en:estonia",
    "en:latvia",
    "en:lithuania",
    "en:luxembourg",
    "en:malta",
    "en:cyprus",
}


def _read_table(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Open Food Facts input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, nrows=max_rows, low_memory=False)
        return df
    elif suffix in {".json", ".jsonl"}:
        df = pd.read_json(path, lines=suffix == ".jsonl")
    else:
        raise ValueError("Unsupported OFF file format. Use parquet, csv, tsv, json, or jsonl.")
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    return df


def _country_group(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).lower()
    tags = {part.strip() for part in text.replace(";", ",").split(",")}
    if tags & EU_MARKERS:
        return "EU"
    if "european union" in text:
        return "EU"
    return "NonEU"


def load_openfoodfacts_uq(
    input_path: str | Path,
    max_rows: int | None = 100_000,
    seed: int = 0,
    mask_eu: float = 0.10,
    mask_noneu: float = 0.35,
) -> dict[str, Any]:
    """Prepare an optional OFF nutrition UQ dataset from a local export."""

    path = Path(input_path)
    raw = _read_table(path, max_rows=max_rows)

    country_col = "countries_tags" if "countries_tags" in raw.columns else "countries_en"
    if country_col not in raw.columns:
        raise ValueError("OFF file must contain countries_tags or countries_en.")

    energy_col = "energy-kcal_100g" if "energy-kcal_100g" in raw.columns else "energy_100g"
    required = ["sugars_100g", "salt_100g", "saturated-fat_100g", energy_col]
    missing_cols = [col for col in required if col not in raw.columns]
    if missing_cols:
        raise ValueError(f"OFF file is missing required nutrition columns: {missing_cols}")

    df = raw[[country_col] + required].copy()
    df["region"] = df[country_col].map(_country_group)
    df = df[df["region"].isin(["EU", "NonEU"])].copy()
    rename = {
        "sugars_100g": "sugars_100g",
        "salt_100g": "salt_100g",
        "saturated-fat_100g": "saturated_fat_100g",
        energy_col: "energy_kcal_100g",
    }
    df = df.rename(columns=rename)
    feature_cols = ["sugars_100g", "salt_100g", "saturated_fat_100g", "energy_kcal_100g"]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["completeness_score"] = df[feature_cols].notna().mean(axis=1)
    df = df[df["completeness_score"] >= 0.5].copy()
    if df.empty:
        raise ValueError("No OFF rows remain after group and completeness filtering.")

    complete = df.dropna(subset=feature_cols).copy()
    if complete.empty:
        raise ValueError("No complete-case rows available for OFF clean evaluation slice.")
    clean_threshold = complete["completeness_score"].quantile(0.75)
    clean_df = complete[complete["completeness_score"] >= clean_threshold][["region"] + feature_cols].copy()

    source = complete[["region"] + feature_cols].copy()
    if len(source) > len(clean_df) * 2:
        source = source.sample(n=min(len(source), max(len(clean_df), 5000)), random_state=seed)
    biased_df = source.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    observed_df = biased_df.copy()
    groups = observed_df["region"].to_numpy()
    miss_probs = np.where(groups == "NonEU", mask_noneu, mask_eu)
    missing = rng.random((len(observed_df), len(feature_cols))) < miss_probs[:, None]
    observed_mask = ~missing
    observed_df.loc[:, feature_cols] = observed_df[feature_cols].mask(missing)

    return {
        "clean_df": clean_df.reset_index(drop=True),
        "biased_df": biased_df.reset_index(drop=True),
        "observed_df": observed_df.reset_index(drop=True),
        "mask": observed_mask,
        "feature_cols": feature_cols,
        "group_col": "region",
        "metadata": {
            "input_path": str(path),
            "max_rows": max_rows,
            "seed": seed,
            "mask_eu": mask_eu,
            "mask_noneu": mask_noneu,
        },
    }

