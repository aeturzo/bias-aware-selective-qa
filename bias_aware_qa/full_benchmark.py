"""Compose a mixed benchmark across all domains for the full experiment."""

from __future__ import annotations

from .benchmark import BiasedAttributeBenchmark
from .config import DOMAIN_PROFILES, ExperimentConfig


def build_mixed(cfg: ExperimentConfig) -> BiasedAttributeBenchmark:
    """Build one benchmark whose domains span every profile in the config."""
    agg = BiasedAttributeBenchmark(n=cfg.n_per_domain)
    agg.domains = []
    for dom_name in cfg.domains:
        if dom_name not in DOMAIN_PROFILES:
            raise ValueError(f"Unknown domain '{dom_name}'. Known: {list(DOMAIN_PROFILES)}")
        prof = DOMAIN_PROFILES[dom_name]
        b = BiasedAttributeBenchmark(n=cfg.n_per_domain, **prof)
        b.build(cfg.seeds, domain_label=dom_name)
        agg.domains.extend(b.domains)
    return agg
