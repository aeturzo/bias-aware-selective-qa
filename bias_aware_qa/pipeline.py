"""Orchestrators: naive evidence-only QA vs bias-aware selective QA.

Both consume a ``BiasedAttributeBenchmark`` and emit a uniform per-query record
(a dict) so the experiment harness can score them with the same metrics.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from dataclasses import replace

from .benchmark import BiasedAttributeBenchmark, Query
from .data_trust import DataTrustLayer, DataTrustConfig, DataTrustResult
from .evidence_qa import EvidenceQA, MockCompassQA, EvidenceAnswer
from .fusion import decide


from .vendor.pdmcmc_uq.metrics import ess


def _domain_profile(domain: str) -> str:
    """Collapse per-seed labels like battery_s0 to battery for tables."""
    head, sep, tail = domain.rpartition("_s")
    return head if sep and tail.isdigit() else domain


def _rank_normalized_split_rhat(values: np.ndarray) -> float:
    """Single-chain split R-hat after rank-normalization.

    This is a lightweight diagnostic for the sampler logs available in this
    package. It is not a substitute for independent multi-chain R-hat, but it
    catches severe nonstationarity in the final chain used for each domain.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    probs = np.clip((ranks - 0.375) / (n + 0.25), 1e-9, 1.0 - 1e-9)
    normal = NormalDist()
    z = np.array([normal.inv_cdf(float(p)) for p in probs], dtype=float)
    if n % 2:
        z = z[1:]
    chains = z.reshape(2, z.size // 2)
    m = chains.shape[1]
    chain_vars = chains.var(axis=1, ddof=1)
    w = float(chain_vars.mean())
    if not np.isfinite(w) or w <= 1e-12:
        return 1.0
    b = float(m * chains.mean(axis=1).var(ddof=1))
    var_hat = ((m - 1.0) / m) * w + b / m
    return float(np.sqrt(max(var_hat / w, 0.0)))


def _record(q: Query, answered: bool, value: float, ci_lo: float, ci_hi: float,
            confidence: float, extra: dict[str, Any]) -> dict[str, Any]:
    correct = bool(answered and np.isfinite(value) and abs(value - q.clean_value) <= q.tolerance)
    covers = bool(answered and np.isfinite(ci_lo) and np.isfinite(ci_hi)
                  and (ci_lo <= q.clean_value <= ci_hi))
    return {
        "qid": q.qid, "domain": q.domain, "domain_profile": _domain_profile(q.domain),
        "seed": q.seed, "quantity": q.quantity, "kind": q.kind,
        "qtype": q.qtype or f"attr_{q.kind}",
        "is_biased": q.is_biased, "is_undercovered": q.is_undercovered,
        "clean_value": q.clean_value, "observed_value": q.observed_value,
        "tolerance": q.tolerance,
        "answered": answered, "value": value,
        "ci_lower": ci_lo, "ci_upper": ci_hi, "ci_covers_clean": covers,
        "confidence": confidence, "correct": correct, **extra,
    }


class NaiveEvidenceQA:
    """Baseline = COMPASS-style evidence-only QA (no data-trust layer).

    Answers the observed (biased) value with evidence-confidence and a naive
    statistical CI; abstains only when evidence is missing.
    """

    def __init__(self, evidence_qa: EvidenceQA | None = None) -> None:
        self.evidence_qa = evidence_qa or MockCompassQA()

    def run(self, benchmark: BiasedAttributeBenchmark) -> list[dict[str, Any]]:
        rows = []
        for q in benchmark.all_queries():
            ev = self.evidence_qa.answer(q)
            answered = ev.has_evidence
            rows.append(_record(
                q, answered, ev.value, ev.stat_ci_lower, ev.stat_ci_upper,
                ev.evidence_confidence if answered else 0.0,
                {"system": "naive", "trust": float("nan")},
            ))
        return rows


class BiasAwareQA:
    """COMPASS evidence layer + latent-bias data-trust layer + fusion."""

    def __init__(
        self,
        evidence_qa: EvidenceQA | None = None,
        data_trust_config: DataTrustConfig | None = None,
        policy_target: dict[str, float] | None = None,
        tau: float = 0.35,
        width_scale: float | None = None,
        sensitivity_scale: float | None = None,
        use_sensitivity: bool = True,
        chain_scale_by_domain: dict[str, float] | None = None,
        memory_reference: Any = None,
    ) -> None:
        self.evidence_qa = evidence_qa or MockCompassQA()
        self.dt_config = data_trust_config or DataTrustConfig()
        self.policy_target = policy_target
        self.tau = tau
        self.width_scale = width_scale
        self.sensitivity_scale = sensitivity_scale
        self.use_sensitivity = use_sensitivity
        self.chain_scale_by_domain = chain_scale_by_domain or {}
        self.memory_reference = memory_reference

    def _config_for(self, domain_name: str) -> DataTrustConfig:
        scale = self.chain_scale_by_domain.get(_domain_profile(domain_name), 1.0)
        if scale == 1.0:
            return self.dt_config
        return replace(self.dt_config,
                       chain_length=int(self.dt_config.chain_length * scale),
                       burn_in=int(self.dt_config.burn_in * scale))

    def run(self, benchmark: BiasedAttributeBenchmark, verbose: bool = False) -> list[dict[str, Any]]:
        # Pass 1: fit the data-trust layer per domain and collect trust signals.
        trust_by_qid: dict[str, DataTrustResult] = {}
        self.diagnostics_ = []
        self.anchor_source_ = None
        for dom in benchmark.domains:
            cfg = self._config_for(dom.name)
            if verbose:
                print(f"  [data-trust] fitting {cfg.n_chains}-chain latent-bias MCMC "
                      f"(len={cfg.chain_length}) for {dom.name} ...", flush=True)
            layer = DataTrustLayer(cfg, self.policy_target, memory_reference=self.memory_reference)
            layer.fit(dom)
            self.anchor_source_ = layer.anchor_source
            diag_quantity = "gap_carbon_kg_per_kwh"
            if diag_quantity not in layer._main_samples or not layer._main_samples.get(diag_quantity, np.array([])).size:
                diag_quantity = next((q.quantity for q in dom.queries if q.kind == "gap"), diag_quantity)
            samples = layer._main_samples.get(diag_quantity, np.array([], dtype=float))
            rhat, mc_ess = layer.chain_diagnostics(diag_quantity)
            self.diagnostics_.append({
                "domain": dom.name,
                "domain_profile": _domain_profile(dom.name),
                "diagnostic_quantity": diag_quantity,
                "n_chains": cfg.n_chains,
                "chain_length": cfg.chain_length,
                "n_samples": int(np.isfinite(samples).sum()),
                "ess_carbon_gap": round(float(mc_ess), 1) if np.isfinite(mc_ess) else float("nan"),
                "rhat_carbon_gap": round(float(rhat), 4) if np.isfinite(rhat) else float("nan"),
                "ess_diagnostic": round(float(mc_ess), 1) if np.isfinite(mc_ess) else float("nan"),
                "rhat_diagnostic": round(float(rhat), 4) if np.isfinite(rhat) else float("nan"),
                "rhat_method": "rank_normalized_multi_chain",
            })
            for q in dom.queries:
                trust_by_qid[q.qid] = layer.query(q)

        # Auto-scale trust to a high percentile of observed uncertainty so only
        # genuinely untrustworthy queries fall below tau (median stays answerable).
        widths = np.array([t.ci_width for t in trust_by_qid.values() if np.isfinite(t.ci_width)])
        sens = np.array([t.target_sensitivity for t in trust_by_qid.values()
                         if np.isfinite(t.target_sensitivity)])
        width_scale = self.width_scale or (float(np.percentile(widths, 75)) if widths.size else 1.0)
        sensitivity_scale = self.sensitivity_scale or (float(np.percentile(sens, 75)) if sens.size else 1.0)
        width_scale = max(width_scale, 1e-6)
        sensitivity_scale = max(sensitivity_scale, 1e-6)
        if not self.use_sensitivity:
            sensitivity_scale = 1e12  # ablation: neutralize the sensitivity channel

        # Pass 2: fuse + decide per query.
        rows = []
        for q in benchmark.all_queries():
            ev = self.evidence_qa.answer(q)
            ts = trust_by_qid[q.qid]
            d = decide(q, ev, ts, self.tau, width_scale, sensitivity_scale, trust_temp=0.5)
            rows.append(_record(
                q, d.answered, d.value, d.ci_lower, d.ci_upper, d.confidence,
                {"system": "bias_aware", "trust": d.trust,
                 "evidence_value": ev.value, "target_sensitivity": ts.target_sensitivity,
                 "ci_width": ts.ci_width},
            ))
        return rows
