"""Bias-Aware Selective QA for Digital Product Passports.

Couples an evidence-grounded QA layer (COMPASS-style: retrieval + memory +
targeted symbolic checks + selective abstention) with a constraint-calibrated
latent-bias MCMC layer (the UQ spinoff) so that the system abstains or corrects
not only when *evidence is missing* but when a recorded attribute value is
*systematically biased*.

This package is self-contained: the UQ MCMC engine is vendored (unchanged) under
``bias_aware_qa.vendor.pdmcmc_uq``; neither of the two original project
codebases is imported or modified at runtime.
"""

from __future__ import annotations

from .benchmark import BiasedAttributeBenchmark, Query
from .data_trust import DataTrustLayer, DataTrustResult, DataTrustConfig
from .evidence_qa import EvidenceQA, MockCompassQA, EvidenceAnswer
from .fusion import BiasAwareDecision, fuse_confidence, decide
from .pipeline import BiasAwareQA, NaiveEvidenceQA
from .config import ExperimentConfig, DOMAIN_PROFILES
from .anchor import AnchorProvider, AttributeAnchor
from .memory_anchor import MemoryCleanReference, CleanFact, seed_memory_from_domains
from . import real_data
from .full_benchmark import build_mixed
from .env_loader import load_dotenv, openai_available
from . import selective

__all__ = [
    "BiasedAttributeBenchmark",
    "Query",
    "DataTrustLayer",
    "DataTrustResult",
    "EvidenceQA",
    "MockCompassQA",
    "EvidenceAnswer",
    "BiasAwareDecision",
    "fuse_confidence",
    "decide",
    "BiasAwareQA",
    "NaiveEvidenceQA",
    "DataTrustConfig",
    "ExperimentConfig",
    "DOMAIN_PROFILES",
    "build_mixed",
    "AnchorProvider",
    "AttributeAnchor",
    "load_dotenv",
    "openai_available",
    "selective",
]

__version__ = "0.1.0"
