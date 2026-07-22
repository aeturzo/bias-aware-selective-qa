"""Additional baselines demanded by a rigorous review.

Two baselines that bracket the contribution of the MCMC layer:

* ``AnchorOnlyQA`` -- returns the *resolved anchor* directly (no MCMC). If the
  full method were merely "reproducing a supplied target", this baseline would
  match it. Differences (interval quality, mean-question handling, abstention
  behavior, robustness under anchor perturbation) isolate what the posterior
  adds beyond the anchor.
* ``apply_matched_coverage`` -- selective RAG at matched coverage: the naive
  evidence-only system, forced to abstain on the same fraction of questions as
  the bias-aware system, by thresholding its own confidence. This tests whether
  the naive system's confidence signal could identify the biased questions if
  only it were allowed to abstain.

Both emit the same per-query record schema as ``pipeline.py`` so the standard
scorer applies unchanged.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np

from .benchmark import BiasedAttributeBenchmark, Query
from .data_trust import (
    DataTrustConfig,
    DataTrustLayer,
    DEFAULT_POLICY_TARGET,
    POLICY_UNCERTAINTY_SD,
)
from .evidence_qa import EvidenceQA, MockCompassQA
from .pipeline import _record


class AnchorOnlyQA:
    """Answers every gap question with the resolved anchor value; no MCMC.

    Anchor resolution (explicit target > memory shrinkage > ontology >
    constants) is *identical* to the full method's -- we reuse
    ``DataTrustLayer.resolve_target`` -- so the comparison isolates the MCMC
    layer, not the anchor plumbing. Mean questions carry no anchor information
    and are answered from the observed records (naive behavior). Intervals are
    the anchor value +/- z * sigma_b with the same bias-uncertainty scale the
    full method uses, so interval-coverage comparisons are apples-to-apples.
    """

    def __init__(self, evidence_qa: EvidenceQA | None = None,
                 data_trust_config: DataTrustConfig | None = None,
                 policy_target: dict[str, float] | None = None,
                 memory_reference: Any = None,
                 confidence: float = 0.9) -> None:
        self.evidence_qa = evidence_qa or MockCompassQA()
        self.cfg = data_trust_config or DataTrustConfig()
        self.policy_target = policy_target
        self.memory_reference = memory_reference
        self.confidence = float(confidence)
        self._z = NormalDist().inv_cdf(0.975)

    def run(self, benchmark: BiasedAttributeBenchmark) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for dom in benchmark.domains:
            layer = DataTrustLayer(self.cfg, self.policy_target,
                                   memory_reference=self.memory_reference)
            target = layer.resolve_target(dom)
            for q in dom.queries:
                ev = self.evidence_qa.answer(q)
                if q.kind == "gap" and q.feature in target:
                    value = float(target[q.feature])
                    extra_sd = POLICY_UNCERTAINTY_SD.get(q.quantity, 0.6 * float(q.tolerance))
                    lo, hi = value - self._z * extra_sd, value + self._z * extra_sd
                    conf = self.confidence
                    answered = True
                else:  # mean questions: the anchor is silent; fall back to evidence
                    value, lo, hi = ev.value, ev.stat_ci_lower, ev.stat_ci_upper
                    conf = ev.evidence_confidence
                    answered = bool(ev.has_evidence)
                rows.append(_record(q, answered, value, lo, hi,
                                    conf if answered else 0.0,
                                    {"system": "anchor_only", "trust": float("nan"),
                                     "anchor_source": layer.anchor_source}))
        return rows


class MAPBaselineQA:
    """MAP baseline: the SAME tilted objective as the full method, optimized
    instead of sampled ("why is MCMC needed?").

    Construction mirrors ``DataTrustLayer`` exactly -- same latent posterior,
    same constraints, same anchor resolution, and the same projected
    dual-ascent lambda calibration -- but replaces sampling with greedy hill
    climbing on log pi_lambda (a proposal is accepted only if it increases the
    objective). The optimum yields a point estimate per query; because there
    is no posterior spread, intervals fall back to the fixed bias-uncertainty
    term (like the anchor-only baseline) and there is no width signal to
    drive abstention, so the baseline answers everything. Anchor sensitivity
    is probed by re-optimizing under the perturbed anchor.
    """

    def __init__(self, evidence_qa: EvidenceQA | None = None,
                 data_trust_config: DataTrustConfig | None = None,
                 policy_target: dict[str, float] | None = None,
                 memory_reference: Any = None,
                 confidence: float = 0.9) -> None:
        self.evidence_qa = evidence_qa or MockCompassQA()
        self.cfg = data_trust_config or DataTrustConfig()
        self.policy_target = policy_target
        self.memory_reference = memory_reference
        self.confidence = float(confidence)
        self._z = NormalDist().inv_cdf(0.975)

    # -- internals ---------------------------------------------------------
    def _optimize(self, domain, target_gap: dict[str, float], seed: int) -> dict[str, float]:
        """Greedy MAP ascent on log p0(z|y) - sum_k lambda_k C_k(z).

        Lambda calibration replicates the sampler's warm-up: after each round,
        projected dual ascent on the mean constraint scores with step
        rho0/sqrt(t+1); lambdas are then frozen for the main ascent.
        """
        from .data_trust import _constraints_factory
        from .vendor.pdmcmc_uq import LatentCleanBiasPosterior
        from .vendor.pdmcmc_uq.dual_update import update_lambdas

        rng = np.random.default_rng(seed)
        posterior = LatentCleanBiasPosterior(
            step_scale=self.cfg.step_scale,
            local_step_scale=self.cfg.local_step_scale,
            bias_step_scale=self.cfg.bias_step_scale,
            obs_noise_scale=self.cfg.obs_noise_scale,
            target_gap=target_gap,
            use_target_gap_in_prior=self.cfg.use_target_gap_in_prior,
        )
        posterior.fit(domain.observed_df, domain.feature_cols, domain.group_col)
        state = posterior.sample_initial_state(domain.observed_df, rng)
        n = len(domain.observed_df)
        rho0_eff = self.cfg.rho0 * max(1.0, n / 500.0)
        # DataTrustLayer resolves epsilon via its anchor; reuse the default.
        constraints = _constraints_factory(
            target_gap, domain.gap_features, domain.feature_cols,
            self.cfg.fairness_epsilon)()

        def objective(s) -> float:
            lp = posterior.log_prob(s)
            for c in constraints:
                lp -= float(c.lambda_value) * float(c.score(s))
            return lp

        cur = objective(state)

        def climb(steps: int, collect: bool = False):
            nonlocal state, cur
            scores = {c.name: [] for c in constraints} if collect else None
            for _ in range(steps):
                ptype = str(rng.choice(["high_bias_cell", "bias_param", "local"]))
                prop, _, _, info = posterior.propose(state, ptype, rng)
                if not info.get("changed", True):
                    continue
                val = objective(prop)
                if val > cur:
                    state, cur = prop, val
                if collect:
                    for c in constraints:
                        scores[c.name].append(float(c.score(state)))
            if collect:
                return {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}

        # warm-up rounds with dual ascent (same recipe as the sampler)
        for t in range(self.cfg.adapt_rounds):
            mean_scores = climb(self.cfg.adapt_steps, collect=True)
            update_lambdas(constraints, mean_scores, t, rho0_eff)
            cur = objective(state)  # lambdas changed -> re-evaluate
        # frozen-lambda main ascent
        climb(self.cfg.chain_length)

        # read off the queried quantities from the MAP state
        out: dict[str, float] = {}
        ref = state.groups == "EU"
        tgt = state.groups == "GS"
        for j, col in enumerate(state.columns):
            out[f"gap_{col}"] = float(state.X_complete[tgt, j].mean()
                                      - state.X_complete[ref, j].mean())
            out[f"mean_{col}"] = float(state.X_complete[:, j].mean())
        return out

    def run(self, benchmark: BiasedAttributeBenchmark) -> list[dict[str, Any]]:
        from .data_trust import SENSITIVITY_DELTA
        rows: list[dict[str, Any]] = []
        for dom in benchmark.domains:
            layer = DataTrustLayer(self.cfg, self.policy_target,
                                   memory_reference=self.memory_reference)
            target = layer.resolve_target(dom)
            est = self._optimize(dom, target, seed=self.cfg.seed + 11)
            alt_target = {c: target.get(c, 0.0) + SENSITIVITY_DELTA.get(c, 0.5)
                          for c in dom.gap_features}
            est_alt = self._optimize(dom, alt_target, seed=self.cfg.seed + 12)
            for q in dom.queries:
                ev = self.evidence_qa.answer(q)
                value = est.get(q.quantity, float("nan"))
                sens = abs(value - est_alt.get(q.quantity, value))
                extra_sd = POLICY_UNCERTAINTY_SD.get(q.quantity, 0.6 * float(q.tolerance))
                lo, hi = value - self._z * extra_sd, value + self._z * extra_sd
                answered = bool(np.isfinite(value)) and ev.has_evidence
                rows.append(_record(q, answered, value, lo, hi,
                                    self.confidence if answered else 0.0,
                                    {"system": "map_baseline", "trust": float("nan"),
                                     "target_sensitivity": sens,
                                     "anchor_source": layer.anchor_source}))
        return rows


def apply_matched_coverage(rows: list[dict[str, Any]], target_coverage: float,
                           seed: int = 0) -> list[dict[str, Any]]:
    """Force a system's coverage down to ``target_coverage`` via its own
    confidence ranking (ties broken randomly but reproducibly).

    Returns new records; abstained rows get answered=False / correct=False.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    k = int(round(target_coverage * n))
    conf = np.array([float(r.get("confidence", 0.0)) if r.get("answered") else -np.inf
                     for r in rows])
    jitter = rng.uniform(0.0, 1e-9, size=n)  # reproducible tie-break only
    order = np.argsort(-(conf + jitter), kind="mergesort")
    keep = set(order[:k].tolist())
    out: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        r2 = dict(r)
        r2["system"] = str(r.get("system", "naive")) + "_matched_cov"
        if i not in keep or not r.get("answered"):
            r2.update({"answered": False, "correct": False,
                       "ci_covers_clean": False})
        out.append(r2)
    return out
