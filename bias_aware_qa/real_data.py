"""Real-world biased-data domains for the bias-aware QA tool.

Three real arms, each turned into ``Domain`` objects the pipeline consumes:

* ``nhanes_domains``      -- REAL-distribution semi-synthetic (controlled truth):
  complete NHANES 2021-2023 rows are clean truth; documented group-dependent
  measurement bias + missingness are injected. Gives rigorous recovery numbers.
* ``openfoodfacts_domains`` -- REAL region-dependent incompleteness (EU vs NonEU)
  over nutrition attributes; the abstention/coverage arm.
* ``climatetrace_domains``  -- REAL in-the-wild reporting bias: self-reported
  national GHG inventories vs an independent Climate TRACE estimate (the anchor).
  Requires a downloaded CSV (see ``ontology``/README); a clearly-labelled
  illustrative sample lets the code run without the download.

All arms relabel their two groups to EU (reference) / GS (target) so the vendored
latent-bias sampler and the anchor machinery apply unchanged. Queries are
group-gap questions per feature (the discriminative quantity).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .benchmark import Domain, Query
from .vendor.pdmcmc_uq.metrics import compute_dataframe_group_gaps


def _relabel(df: pd.DataFrame, group_col: str, ref: str, target: str) -> pd.DataFrame:
    out = df.copy()
    out["region"] = out[group_col].map({ref: "EU", target: "GS"})
    return out[out["region"].isin(["EU", "GS"])]


def domain_from_frames(name: str, seed: int, clean_df: pd.DataFrame, observed_df: pd.DataFrame,
                       feature_cols: list[str], group_col: str, ref: str, target: str,
                       gap_tol_frac: float = 0.25, min_group_evidence: int = 20) -> Domain:
    """Build a Domain (gap queries per feature) from a loader's clean/observed frames."""
    clean = _relabel(clean_df, group_col, ref, target)
    obs = _relabel(observed_df, group_col, ref, target)
    dom = Domain(name=name, seed=seed, clean_df=clean, observed_df=obs,
                 feature_cols=list(feature_cols), group_col="region",
                 gap_features=list(feature_cols))
    clean_gaps = compute_dataframe_group_gaps(clean, feature_cols, group_col="region")
    for feat in feature_cols:
        clean_v = float(clean_gaps[feat])
        sub = obs[["region", feat]].dropna()
        eu, gs = sub[sub.region == "EU"][feat], sub[sub.region == "GS"][feat]
        n_eu, n_gs = len(eu), len(gs)
        obs_v = float(gs.mean() - eu.mean()) if n_eu and n_gs else float("nan")
        scale = max(abs(clean_v), float(clean[feat].std()), 1e-6)
        tol = gap_tol_frac * scale
        biased = (not np.isnan(obs_v)) and abs(obs_v - clean_v) > tol
        undercov = n_eu < min_group_evidence or n_gs < min_group_evidence
        dom.queries.append(Query(
            qid=f"{name}:gap:{feat}", domain=name, seed=seed, quantity=f"gap_{feat}",
            feature=feat, kind="gap", clean_value=clean_v, observed_value=obs_v,
            n_evidence_eu=n_eu, n_evidence_gs=n_gs, is_biased=biased,
            is_undercovered=undercov, tolerance=tol, qtype="attr_gap"))
    return dom


# ---------------------------------------------------------------- NHANES
def nhanes_domains(data_dir: str | Path, task: str = "cardiometabolic_race",
                   seeds: Iterable[int] = (0, 1, 2), max_rows: int = 1500,
                   bias_strength: float = 1.0) -> list[Domain]:
    from .vendor.pdmcmc_uq.nhanes_data import prepare_nhanes_uq
    doms = []
    for s in seeds:
        d = prepare_nhanes_uq(data_dir, task=task, max_rows=max_rows, seed=int(s),
                              bias_strength=bias_strength)
        gcol = d.get("group_col", "race_ethnicity")
        feats = d.get("feature_cols") or [c for c in d["clean_df"].columns if c != gcol]
        doms.append(domain_from_frames(
            f"nhanes_{task}_s{s}", int(s), d["clean_df"], d["observed_df"],
            list(feats), gcol, d["ref_group"], d["target_group"]))
    return doms


# ---------------------------------------------------------------- Open Food Facts
def openfoodfacts_domains(csv_path: str | Path, seeds: Iterable[int] = (0, 1, 2),
                          max_rows: int = 60000) -> list[Domain]:
    from .vendor.pdmcmc_uq.openfoodfacts_data import load_openfoodfacts_uq
    path = Path(csv_path)
    src = path
    if path.suffix.lower() == ".gz":
        # OFF export is a gzipped TSV; decompress a header+sample slice to .tsv
        import gzip
        import tempfile
        src = Path(tempfile.gettempdir()) / f"off_sample_{max_rows}.tsv"
        if not src.exists():
            with gzip.open(path, "rt", errors="replace") as fin, src.open("w") as fout:
                for i, line in enumerate(fin):
                    fout.write(line)
                    if i >= max_rows:
                        break
    doms = []
    for s in seeds:
        d = load_openfoodfacts_uq(src, max_rows=max_rows, seed=int(s))
        doms.append(domain_from_frames(
            f"off_real_s{s}", int(s), d["clean_df"], d["observed_df"],
            list(d["feature_cols"]), d.get("group_col", "region"), "EU", "NonEU"))
    return doms


# ---------------------------------------------------------------- Climate TRACE
# Real download: national self-reported inventories (EEA/UNFCCC or EDGAR) joined
# with Climate TRACE country totals. Required columns:
#   country, group(AnnexI|nonAnnexI), self_reported, climatetrace
# Extra metadata columns such as unit/source/feature are preserved in the CSV
# but ignored by the domain builder.
# The self-reported value is the biased "observed"; Climate TRACE is the anchor
# (independent reference), so we treat Climate TRACE as the clean reference.
_CT_SAMPLE = Path(__file__).resolve().parents[1] / "real_data" / "climatetrace_sample.csv"


def climatetrace_domains(csv_path: str | Path | None = None,
                         seeds: Iterable[int] = (0,), feature: str = "emissions_mtco2e") -> list[Domain]:
    path = Path(csv_path) if csv_path else _CT_SAMPLE
    raw = pd.read_csv(path)
    required = {"country", "group", "self_reported", "climatetrace"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Climate TRACE CSV missing required columns: {sorted(missing)}")
    if feature == "emissions_mtco2e" and "feature" in raw.columns:
        features = [v for v in raw["feature"].dropna().astype(str).unique() if v]
        if len(features) == 1:
            feature = features[0]
    raw["self_reported"] = pd.to_numeric(raw["self_reported"], errors="coerce")
    raw["climatetrace"] = pd.to_numeric(raw["climatetrace"], errors="coerce")
    raw = raw.dropna(subset=["group", "self_reported", "climatetrace"])
    # long form: one row per country with self_reported + climatetrace
    ref_grp = raw[raw["group"] == "AnnexI"]
    tgt_grp = raw[raw["group"] == "nonAnnexI"]
    if ref_grp.empty or tgt_grp.empty:
        raise ValueError("Climate TRACE CSV must contain both AnnexI and nonAnnexI groups")
    doms = []
    for s in seeds:
        # clean = Climate TRACE (independent reference); observed = self-reported
        clean = pd.concat([
            pd.DataFrame({"grp": "AnnexI", feature: ref_grp["climatetrace"]}),
            pd.DataFrame({"grp": "nonAnnexI", feature: tgt_grp["climatetrace"]}),
        ], ignore_index=True)
        obs = pd.concat([
            pd.DataFrame({"grp": "AnnexI", feature: ref_grp["self_reported"]}),
            pd.DataFrame({"grp": "nonAnnexI", feature: tgt_grp["self_reported"]}),
        ], ignore_index=True)
        doms.append(domain_from_frames(
            f"climatetrace_s{s}", int(s), clean, obs, [feature], "grp",
            "AnnexI", "nonAnnexI", min_group_evidence=5))
    return doms
