"""Biased-attribute QA benchmark over DPP-style sustainability data.

Each *query* asks for a group-level quantity (a GS-EU disparity for a
sustainability attribute, or a mean attribute value). We know:

  * the CLEAN answer (ground truth, from the clean reference table),
  * the OBSERVED answer that an evidence-grounded QA layer would report from the
    biased/incomplete records (this is what a naive RAG/LLM returns),
  * whether the query is "biased" (observed answer materially deviates from
    clean) and/or "under-covered" (too few observed records to answer).

The benchmark is built with the vendored ``generate_synthetic_dpp_uq`` so that
clean/biased/observed tables are internally consistent. Real DPP corpora and
Open Food Facts can be substituted by supplying (clean_df, observed_df).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .vendor.pdmcmc_uq import generate_synthetic_dpp_uq
from .vendor.pdmcmc_uq.metrics import compute_dataframe_group_gaps

# Attributes for which we form GS-EU gap queries, plus mean-value queries.
GAP_FEATURES = ["carbon_kg_per_kwh", "repairability_score", "durability_score"]
MEAN_FEATURES = ["carbon_kg_per_kwh"]


@dataclass
class Query:
    """A single benchmark question."""

    qid: str
    domain: str
    seed: int
    quantity: str            # e.g. "gap_carbon_kg_per_kwh" or "mean_carbon_kg_per_kwh"
    feature: str
    kind: str                # "gap" | "mean"
    clean_value: float       # ground-truth answer
    observed_value: float    # what the biased/incomplete records imply
    n_evidence_eu: int
    n_evidence_gs: int
    is_biased: bool          # observed materially deviates from clean
    is_undercovered: bool    # too little evidence to answer safely
    tolerance: float         # |answer - clean| <= tolerance counts as correct
    qtype: str = ""          # unified question type, e.g. "attr_gap" / "attr_mean"


@dataclass
class Domain:
    """A domain instance: clean/observed tables + generated queries."""

    name: str
    seed: int
    clean_df: pd.DataFrame
    observed_df: pd.DataFrame
    feature_cols: list[str]
    group_col: str
    queries: list[Query] = field(default_factory=list)
    # numeric features that carry a group-gap query (defaults to the DPP set;
    # real-data domains set their own feature list)
    gap_features: list[str] = field(default_factory=lambda: list(GAP_FEATURES))


def _observed_gap(observed_df: pd.DataFrame, feature: str, group_col: str) -> tuple[float, int, int]:
    """GS-EU gap computed only over observed (non-missing) records."""
    sub = observed_df[[group_col, feature]].dropna()
    eu = sub[sub[group_col] == "EU"][feature]
    gs = sub[sub[group_col] == "GS"][feature]
    if len(eu) == 0 or len(gs) == 0:
        return float("nan"), len(eu), len(gs)
    return float(gs.mean() - eu.mean()), len(eu), len(gs)


def _observed_mean(observed_df: pd.DataFrame, feature: str) -> tuple[float, int]:
    col = observed_df[feature].dropna()
    if len(col) == 0:
        return float("nan"), 0
    return float(col.mean()), len(col)


class BiasedAttributeBenchmark:
    """Builds and holds a multi-domain biased-attribute QA benchmark."""

    def __init__(
        self,
        n: int = 800,
        gs_frac: float = 0.4,
        bias_strength: float = 1.0,
        missing_eu: float = 0.10,
        missing_gs: float = 0.35,
        gap_tol_frac: float = 0.25,
        min_group_evidence: int = 40,
    ) -> None:
        self.n = n
        self.gs_frac = gs_frac
        self.bias_strength = bias_strength
        self.missing_eu = missing_eu
        self.missing_gs = missing_gs
        self.gap_tol_frac = gap_tol_frac
        self.min_group_evidence = min_group_evidence
        self.domains: list[Domain] = []

    def build(self, seeds: list[int], domain_label: str | None = None) -> "BiasedAttributeBenchmark":
        self.domains = []
        for seed in seeds:
            data = generate_synthetic_dpp_uq(
                n=self.n,
                gs_frac=self.gs_frac,
                bias_strength=self.bias_strength,
                missing_eu=self.missing_eu,
                missing_gs=self.missing_gs,
                seed=seed,
            )
            clean_df = data["clean_df"]
            observed_df = data["observed_df"]
            group_col = data["group_col"]
            feature_cols = data["feature_cols"]
            clean_gaps = compute_dataframe_group_gaps(clean_df, GAP_FEATURES, group_col=group_col)

            base = domain_label or "synthetic_dpp"
            dom = Domain(
                name=f"{base}_s{seed}",
                seed=seed,
                clean_df=clean_df,
                observed_df=observed_df,
                feature_cols=feature_cols,
                group_col=group_col,
            )
            # gap queries
            for feat in GAP_FEATURES:
                clean_v = float(clean_gaps[feat])
                obs_v, n_eu, n_gs = _observed_gap(observed_df, feat, group_col)
                scale = max(abs(clean_v), float(clean_df[feat].std()), 1e-6)
                tol = self.gap_tol_frac * scale
                undercov = (n_eu < self.min_group_evidence) or (n_gs < self.min_group_evidence)
                biased = (not np.isnan(obs_v)) and (abs(obs_v - clean_v) > tol)
                dom.queries.append(
                    Query(
                        qid=f"{dom.name}:gap:{feat}",
                        domain=dom.name,
                        seed=seed,
                        quantity=f"gap_{feat}",
                        feature=feat,
                        kind="gap",
                        clean_value=clean_v,
                        observed_value=obs_v,
                        n_evidence_eu=n_eu,
                        n_evidence_gs=n_gs,
                        is_biased=biased,
                        is_undercovered=undercov,
                        tolerance=tol,
                        qtype="attr_gap",
                    )
                )
            # mean queries
            for feat in MEAN_FEATURES:
                clean_v = float(clean_df[feat].mean())
                obs_v, n_obs = _observed_mean(observed_df, feat)
                scale = max(abs(clean_v), float(clean_df[feat].std()), 1e-6)
                tol = self.gap_tol_frac * scale
                undercov = n_obs < 2 * self.min_group_evidence
                biased = (not np.isnan(obs_v)) and (abs(obs_v - clean_v) > tol)
                dom.queries.append(
                    Query(
                        qid=f"{dom.name}:mean:{feat}",
                        domain=dom.name,
                        seed=seed,
                        quantity=f"mean_{feat}",
                        feature=feat,
                        kind="mean",
                        clean_value=clean_v,
                        observed_value=obs_v,
                        n_evidence_eu=n_obs,
                        n_evidence_gs=n_obs,
                        is_biased=biased,
                        is_undercovered=undercov,
                        tolerance=tol,
                        qtype="attr_mean",
                    )
                )
            self.domains.append(dom)
        return self

    def all_queries(self) -> list[Query]:
        return [q for d in self.domains for q in d.queries]

    def summary(self) -> dict[str, Any]:
        qs = self.all_queries()
        return {
            "n_domains": len(self.domains),
            "n_queries": len(qs),
            "n_biased": sum(q.is_biased for q in qs),
            "n_undercovered": sum(q.is_undercovered for q in qs),
        }
