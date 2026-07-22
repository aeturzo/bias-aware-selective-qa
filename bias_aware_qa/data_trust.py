"""Data-trust layer: latent-bias uncertainty over queried attributes.

This is the novel coupling. For a domain's (biased, incomplete) observed table
we run the constraint-calibrated latent-bias MCMC (vendored UQ engine) once,
obtaining posterior samples of every group-gap / mean quantity. For each query
we then read off:

  * ``x_clean_hat`` -- the debiased posterior mean (the corrected answer),
  * a credible interval and its width (uncertainty),
  * ``target_sensitivity`` -- how much the debiased estimate moves when the
    auditable fairness/policy target is perturbed (identifiability signal).

These signals feed the bias-aware confidence fusion and abstention rule.

The MCMC never sees the clean reference; ``clean_value`` is used only to
evaluate interval coverage after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import Domain, Query, GAP_FEATURES
from .vendor.pdmcmc_uq import (
    LatentCleanBiasPosterior,
    SamplerConfig,
    FairnessGapConstraint,
    BiasPenaltyConstraint,
    CoverageConstraint,
)
from .vendor.pdmcmc_uq.baselines import run_pdmcmc_method
from .vendor.pdmcmc_uq.uq_intervals import interval_summary


# Auditable domain-level policy target for the expected group disparity (an
# expert/regulatory *assumption*, fixed per domain -- NOT the per-instance clean
# value used for evaluation). Matches the UQ study's synthetic policy target.
DEFAULT_POLICY_TARGET = {
    "carbon_kg_per_kwh": 5.0,
    "repairability_score": -0.30,
    "durability_score": -0.40,
}
# Perturbation used to probe target-sensitivity (per-feature).
SENSITIVITY_DELTA = {
    "carbon_kg_per_kwh": 2.0,
    "repairability_score": 0.6,
    "durability_score": 0.6,
}
# Bias-uncertainty added to intervals (policy/measurement uncertainty, not just
# sampling noise) so credible intervals honestly reflect measurement bias.
POLICY_UNCERTAINTY_SD = {
    "gap_carbon_kg_per_kwh": 0.75,
    "gap_repairability_score": 0.15,
    "gap_durability_score": 0.15,
    "mean_carbon_kg_per_kwh": 0.35,
}


@dataclass
class DataTrustResult:
    quantity: str
    x_clean_hat: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    target_sensitivity: float
    covers_clean: bool
    n_samples: int


@dataclass
class DataTrustConfig:
    """Latent-bias MCMC settings.

    Defaults replicate the UQ study's ``latent_adaptive_single_agent`` recipe,
    which is what achieves clean-gap recovery with full interval coverage.
    ``rho0`` is the base step/temperature scale and is multiplied by
    ``max(1, n/500)`` per the study; the large value is intentional.
    """

    chain_length: int = 1500
    burn_in: int = 400
    adapt_rounds: int = 4
    adapt_steps: int = 200
    rho0: float = 300.0
    proposal_type: str = "high_bias_cell"
    method: str = "adaptive_single_agent"
    fairness_epsilon: float = 0.35
    # latent-posterior step/noise scales (UQ defaults)
    step_scale: float = 0.18
    local_step_scale: float = 0.06
    bias_step_scale: float = 0.08
    obs_noise_scale: float = 0.20
    use_target_gap_in_prior: bool = True
    constrained: bool = True   # ablation: False -> unconstrained latent (no fairness pull)
    n_chains: int = 4          # independent chains for pooled intervals + R-hat
    memory_prior_pseudocount: float = 12.0  # shrinkage: audit facts vs ontology prior
    # anchor-robustness probes (misspecification experiments; 1.0/False = off)
    anchor_scale: float = 1.0        # multiply the resolved gap anchor by this
    anchor_flip_sign: bool = False   # wrong-sign anchor stress test
    seed: int = 0


def _constraints_factory(target_gap: dict[str, float], gap_features: list[str],
                         all_features: list[str], eps: float):
    gap_cols = list(gap_features)

    def factory() -> list:
        return [
            FairnessGapConstraint(
                feature_cols=gap_cols,
                epsilon=eps,
                lambda_value=0.0,
                target_gap={c: target_gap.get(c, 0.0) for c in gap_cols},
                normalize=True,
            ),
            BiasPenaltyConstraint(
                feature_cols=all_features, epsilon=0.75, lambda_value=0.0,
                target_gap={c: target_gap.get(c, 0.0) for c in gap_cols},
            ),
            CoverageConstraint(epsilon=0.0, lambda_value=0.0),
        ]

    return factory


def _log_column(quantity: str) -> str:
    if quantity.startswith("mean_"):
        return "mean_carbon"  # only carbon mean is logged by the engine
    return quantity  # gap_<feature> matches the logged column name


class DataTrustLayer:
    """Fits the latent-bias posterior for a domain and answers queries."""

    def __init__(self, config: DataTrustConfig | None = None,
                 policy_target: dict[str, float] | None = None,
                 anchor: "AnchorProvider | None" = None,
                 memory_reference: "MemoryCleanReference | None" = None) -> None:
        from .anchor import AnchorProvider
        self.cfg = config or DataTrustConfig()
        # Anchor resolution order (strongest first):
        #   explicit arg  >  external-memory verified clean facts (measured truth)
        #                 >  DPP ontology triples (auditable assumption)
        #                 >  built-in constants.
        self.anchor = anchor or AnchorProvider.from_ontology()
        self.memory_reference = memory_reference
        self._explicit_target = dict(policy_target) if policy_target else None
        self._base_target = dict(self._explicit_target or self.anchor.policy_target() or DEFAULT_POLICY_TARGET)
        self.policy_target = dict(self._base_target)  # may be refined per-domain in fit()
        self._fairness_eps = float(self.anchor.epsilon(self.cfg.fairness_epsilon))
        self.anchor_source = self.anchor.source
        self._main_samples: dict[str, np.ndarray] = {}
        self._main_chains: dict[str, list[np.ndarray]] = {}
        self._alt_means: dict[str, float] = {}
        self._fitted = False

    def _resolve_target(self, domain: Domain) -> dict[str, float]:
        """Per-domain policy target.

        When verified clean facts exist in external memory, the target for a
        feature is a shrinkage blend of the audited gap and the ontology prior,
        weighted by the number of audited facts:
            target = (n * audit_gap + k * prior) / (n + k)
        so a tiny/noisy audit stays close to the ontology prior, while a large
        audit is trusted as measured truth. Falls back to ontology/constants.
        """
        target = dict(self._base_target)
        if self.memory_reference is not None and self._explicit_target is None:
            head, sep, tail = domain.name.rpartition("_s")
            family = head if sep and tail.isdigit() else domain.name
            k = float(self.cfg.memory_prior_pseudocount)
            used = 0
            for feat in domain.gap_features:
                gap, n = self.memory_reference.group_gap_with_n(feat, domain=family)
                if gap is None:
                    gap, n = self.memory_reference.group_gap_with_n(feat, domain=None)
                if gap is None:
                    continue
                prior = float(self._base_target.get(feat, 0.0))
                target[feat] = (n * gap + k * prior) / (n + k)
                used += 1
            if used:
                self.anchor_source = (f"{self.memory_reference.source} "
                                      f"[{used} feat memory+prior shrinkage k={k:g}]")
        # anchor-misspecification probes (robustness experiments)
        scale = float(self.cfg.anchor_scale)
        flip = bool(self.cfg.anchor_flip_sign)
        if scale != 1.0 or flip:
            sgn = -1.0 if flip else 1.0
            for feat in domain.gap_features:
                if feat in target:
                    target[feat] = sgn * scale * float(target[feat])
            self.anchor_source = (f"{self.anchor_source} "
                                  f"[perturbed scale={scale:g} flip={flip}]")
        return target

    def resolve_target(self, domain: Domain) -> dict[str, float]:
        """Public anchor resolution (used by anchor-only baseline & audits)."""
        return self._resolve_target(domain)

    def _run(self, domain: Domain, target_gap: dict[str, float], seed_offset: int) -> pd.DataFrame:
        constrained = self.cfg.constrained
        posterior = LatentCleanBiasPosterior(
            step_scale=self.cfg.step_scale,
            local_step_scale=self.cfg.local_step_scale,
            bias_step_scale=self.cfg.bias_step_scale,
            obs_noise_scale=self.cfg.obs_noise_scale,
            target_gap=target_gap,
            use_target_gap_in_prior=self.cfg.use_target_gap_in_prior and constrained,
        )
        posterior.fit(domain.observed_df, domain.feature_cols, domain.group_col)
        n = len(domain.observed_df)
        rho0_eff = self.cfg.rho0 * max(1.0, n / 500.0)
        method = self.cfg.method if constrained else "fixed_lambda"
        init_lambdas = None if constrained else {"fairness_gap": 0.0, "bias_penalty": 0.0}
        config = SamplerConfig(
            method=method,
            adapt_rounds=self.cfg.adapt_rounds,
            adapt_steps=self.cfg.adapt_steps,
            chain_length=self.cfg.chain_length,
            burn_in=self.cfg.burn_in,
            rho0=rho0_eff,
            proposal_type=self.cfg.proposal_type,
            seed=self.cfg.seed + seed_offset,
            store_samples=False,
        )
        result = run_pdmcmc_method(
            method,
            posterior,
            _constraints_factory(target_gap, domain.gap_features, domain.feature_cols, self._fairness_eps),
            domain.observed_df,
            domain.feature_cols,
            domain.group_col,
            config,
            seed=self.cfg.seed + seed_offset,
            clean_df=None,
            initial_lambdas=init_lambdas,
        )
        logs = result.logs
        return logs[logs["phase"] == "sample"] if "phase" in logs else logs

    def fit(self, domain: Domain) -> "DataTrustLayer":
        # resolve the anchor for this domain (memory-derived target if available)
        self.policy_target = self._resolve_target(domain)
        # main posterior under the policy target: run n_chains independent chains
        self._main_chains = {q.quantity: [] for q in domain.queries}
        for c in range(max(1, self.cfg.n_chains)):
            df = self._run(domain, self.policy_target, seed_offset=100 * (c + 1))
            for q in domain.queries:
                col = _log_column(q.quantity)
                if col in df:
                    self._main_chains[q.quantity].append(df[col].to_numpy(dtype=float))
        # pooled samples for intervals / point estimate
        self._main_samples = {
            qn: (np.concatenate(arrs) if arrs else np.array([], dtype=float))
            for qn, arrs in self._main_chains.items()
        }
        # alternative target (single chain) to probe policy/target sensitivity
        alt_target = {
            c: self.policy_target.get(c, 0.0) + SENSITIVITY_DELTA.get(c, 0.5)
            for c in domain.gap_features
        }
        alt = self._run(domain, alt_target, seed_offset=777)
        self._alt_means = {}
        for q in domain.queries:
            col = _log_column(q.quantity)
            if col in alt:
                self._alt_means[q.quantity] = float(np.nanmean(alt[col].to_numpy(dtype=float)))
        self._fitted = True
        return self

    def chain_diagnostics(self, quantity: str) -> tuple[float, float]:
        """(rank-normalized multi-chain R-hat, multi-chain ESS) for a quantity."""
        from .vendor.pdmcmc_uq.diagnostics import rank_normalized_rhat, multi_chain_ess
        from .vendor.pdmcmc_uq.metrics import ess as single_ess
        chains = [c for c in self._main_chains.get(quantity, []) if c.size > 1]
        if len(chains) >= 2:
            try:
                return float(rank_normalized_rhat(chains)), float(multi_chain_ess(chains))
            except Exception:
                pass
        pooled = self._main_samples.get(quantity, np.array([]))
        return float("nan"), float(single_ess(pooled)) if pooled.size else float("nan")

    def query(self, q: Query) -> DataTrustResult:
        if not self._fitted:
            raise RuntimeError("DataTrustLayer.fit(domain) must be called first.")
        samples = self._main_samples.get(q.quantity, np.array([], dtype=float))
        # bias/measurement uncertainty added to the interval: explicit per-DPP-quantity
        # value if declared, else a generic term scaled to the attribute tolerance
        # (so real-data features get honestly-widened intervals too).
        extra_sd = POLICY_UNCERTAINTY_SD.get(q.quantity)
        if extra_sd is None:
            extra_sd = 0.6 * float(q.tolerance)
        summ = interval_summary(samples, q.clean_value, alpha=0.05, extra_sd=extra_sd)
        mean = summ["posterior_mean"]
        alt_mean = self._alt_means.get(q.quantity, mean)
        sensitivity = float(abs(mean - alt_mean)) if np.isfinite(mean) and np.isfinite(alt_mean) else float("inf")
        return DataTrustResult(
            quantity=q.quantity,
            x_clean_hat=float(mean),
            ci_lower=float(summ["ci_lower"]),
            ci_upper=float(summ["ci_upper"]),
            ci_width=float(summ["ci_width"]),
            target_sensitivity=sensitivity,
            covers_clean=bool(summ["ci_covers_clean"] >= 0.5),
            n_samples=int(np.isfinite(samples).sum()),
        )
