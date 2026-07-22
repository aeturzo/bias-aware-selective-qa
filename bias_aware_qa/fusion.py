"""Bias-aware confidence fusion and abstention.

The evidence layer yields ``c_ev`` (grounding confidence). The data-trust layer
yields a normalized uncertainty (credible-interval width) and a target
sensitivity. We fuse them into a single monotone confidence

    c = c_ev * trust,   trust = exp( -(w / w0 + s / s0) ),

which is monotone-decreasing in both interval width ``w`` and target
sensitivity ``s`` -- so the system is confident only when evidence is present
*and* the debiased value is tight and target-insensitive. The abstention rule is
then the familiar selective-prediction threshold ``c < tau``, but now it fires
on *untrustworthy-but-present* evidence, not only on missing evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .benchmark import Query
from .data_trust import DataTrustResult
from .evidence_qa import EvidenceAnswer


@dataclass
class BiasAwareDecision:
    qid: str
    answered: bool
    value: float                 # reported (debiased) value if answered
    ci_lower: float
    ci_upper: float
    confidence: float
    trust: float
    provenance: dict[str, Any] = field(default_factory=dict)


def fuse_confidence(
    evidence: EvidenceAnswer,
    trust_signal: DataTrustResult,
    width_scale: float,
    sensitivity_scale: float,
    trust_temp: float = 0.5,
) -> tuple[float, float]:
    """Return (fused_confidence, trust) in [0, 1].

    ``trust_temp`` softens the penalty so a typical (median-uncertainty) query is
    still answerable; the scales are set to a high percentile of the observed
    uncertainty so that only genuinely untrustworthy queries fall below tau.
    """
    if not evidence.has_evidence:
        return 0.0, 0.0
    w = trust_signal.ci_width if np.isfinite(trust_signal.ci_width) else np.inf
    s = trust_signal.target_sensitivity if np.isfinite(trust_signal.target_sensitivity) else np.inf
    penalty = trust_temp * (w / max(width_scale, 1e-9) + s / max(sensitivity_scale, 1e-9))
    trust = float(np.clip(np.exp(-penalty), 0.0, 1.0))
    fused = float(np.clip(evidence.evidence_confidence * trust, 0.0, 1.0))
    return fused, trust


def decide(
    query: Query,
    evidence: EvidenceAnswer,
    trust_signal: DataTrustResult,
    tau: float,
    width_scale: float,
    sensitivity_scale: float,
    trust_temp: float = 0.5,
) -> BiasAwareDecision:
    """Bias-aware answer/abstain decision with a debiased, interval answer."""
    conf, trust = fuse_confidence(evidence, trust_signal, width_scale, sensitivity_scale, trust_temp)
    answered = conf >= tau
    return BiasAwareDecision(
        qid=query.qid,
        answered=answered,
        value=float(trust_signal.x_clean_hat),          # debiased posterior mean
        ci_lower=float(trust_signal.ci_lower),
        ci_upper=float(trust_signal.ci_upper),
        confidence=conf,
        trust=trust,
        provenance={
            "evidence_value": evidence.value,            # what the biased records said
            "evidence_confidence": evidence.evidence_confidence,
            "n_evidence": evidence.n_evidence,
            "ci_width": trust_signal.ci_width,
            "target_sensitivity": trust_signal.target_sensitivity,
            "quantity": query.quantity,
        },
    )
