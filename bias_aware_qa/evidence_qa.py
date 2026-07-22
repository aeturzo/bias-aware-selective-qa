"""Evidence-grounded QA layer (COMPASS stand-in).

In deployment this wraps COMPASS: hybrid retrieval + persistent memory +
targeted symbolic checks produce an answer, an evidence-confidence, and an
abstain flag when evidence is missing. For a self-contained, API-free
experiment we provide ``MockCompassQA``, which reproduces COMPASS's *relevant*
behaviour for attribute queries: it answers the value implied by the observed
(biased/incomplete) records with a confidence that grows with evidence
coverage, and abstains only when coverage is too low.

The key point the experiment demonstrates is that an evidence-grounded layer,
however strong, returns the *observed* value -- which is confidently biased when
the underlying records are systematically biased. The data-trust layer
(``data_trust.py``) is what corrects this.

To plug in the real COMPASS, implement the ``EvidenceQA`` protocol against
``backend/api/answerer_ctx.py`` / ``backend/services/confidence.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .benchmark import Query


@dataclass
class EvidenceAnswer:
    value: float                 # answer implied by observed evidence
    evidence_confidence: float   # in [0, 1]
    has_evidence: bool           # False -> COMPASS abstains (missing evidence)
    stat_ci_lower: float         # naive statistical CI around the observed value
    stat_ci_upper: float
    n_evidence: int


class EvidenceQA(Protocol):
    def answer(self, query: Query) -> EvidenceAnswer: ...


class MockCompassQA:
    """A faithful stand-in for the evidence layer on attribute queries."""

    def __init__(self, min_group_evidence: int = 40, high_conf: float = 0.9) -> None:
        self.min_group_evidence = min_group_evidence
        self.high_conf = high_conf

    def answer(self, query: Query) -> EvidenceAnswer:
        obs = query.observed_value
        n_eff = min(query.n_evidence_eu, query.n_evidence_gs) if query.kind == "gap" else query.n_evidence_eu

        # COMPASS abstains when evidence is missing / too sparse.
        if np.isnan(obs) or n_eff <= 0:
            return EvidenceAnswer(float("nan"), 0.0, False, float("nan"), float("nan"), int(max(n_eff, 0)))

        # Evidence-confidence grows with coverage and saturates high -- this is
        # exactly why a strong RAG/LLM is *confidently* wrong on biased data.
        cov_ratio = min(1.0, n_eff / (2.0 * self.min_group_evidence))
        evidence_confidence = float(self.high_conf * (0.6 + 0.4 * cov_ratio))
        # The benchmark/domain builder owns the evidence sufficiency threshold:
        # synthetic DPP queries use 40/group, while compact real arms such as
        # Climate TRACE deliberately use a smaller country-count threshold.
        has_evidence = bool((not query.is_undercovered) and n_eff > 0)

        # Naive statistical CI: standard error of the observed estimate only
        # (captures sampling noise, NOT measurement bias).
        se = query.tolerance / 2.0 / max(np.sqrt(max(n_eff, 1)), 1.0) * np.sqrt(max(n_eff, 1))
        se = query.tolerance * 0.6  # width comparable to tolerance; bias-blind
        return EvidenceAnswer(
            value=float(obs),
            evidence_confidence=evidence_confidence if has_evidence else 0.2,
            has_evidence=has_evidence,
            stat_ci_lower=float(obs - se),
            stat_ci_upper=float(obs + se),
            n_evidence=int(n_eff),
        )
