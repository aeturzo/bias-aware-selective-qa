"""Experiment configuration: domain profiles and run sizes.

The four domains mirror the ICTAI paper's coverage. For the numeric
attribute/gap questions we use structurally-realistic synthetic profiles with
per-domain bias/missingness characteristics; the Open Food Facts profile is the
most incomplete (its sustainability fields are ~83% missing in reality). If a
real Open Food Facts CSV is available, ``load_off_domain`` uses it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Per-domain generator characteristics (bias strength, missingness, GS fraction).
DOMAIN_PROFILES: dict[str, dict[str, float]] = {
    "battery":     {"bias_strength": 1.3, "missing_eu": 0.10, "missing_gs": 0.35, "gs_frac": 0.40},
    "lexmark":     {"bias_strength": 1.0, "missing_eu": 0.08, "missing_gs": 0.30, "gs_frac": 0.35},
    "viessmann":   {"bias_strength": 1.5, "missing_eu": 0.12, "missing_gs": 0.45, "gs_frac": 0.45},
    "openfoodfacts": {"bias_strength": 1.2, "missing_eu": 0.20, "missing_gs": 0.55, "gs_frac": 0.40},
}

# Harder domains mix more slowly; scale their chain length so multi-chain
# rank-normalized R-hat reaches <= 1.05. Others keep the base length.
CHAIN_SCALE: dict[str, float] = {
    "battery": 1.0, "lexmark": 1.0, "openfoodfacts": 1.0, "viessmann": 3.0,
}


@dataclass
class ExperimentConfig:
    domains: list[str] = field(default_factory=lambda: list(DOMAIN_PROFILES))
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_per_domain: int = 1200
    # MCMC (data-trust) sizes
    chain_length: int = 1500
    burn_in: int = 400
    adapt_rounds: int = 4
    adapt_steps: int = 200
    n_chains: int = 4
    tau: float = 0.35
    # real components
    use_compass: bool = False           # use real COMPASS answerer for evidence
    off_csv: str | None = None          # path to real Open Food Facts products csv
    compass_eval_csvs: list[str] = field(default_factory=list)  # COMPASS text-QA results to merge

    def smoke(self) -> "ExperimentConfig":
        """Small, fast settings for a smoke test."""
        self.seeds = [0]
        self.n_per_domain = 400
        self.chain_length = 500
        self.burn_in = 150
        self.adapt_rounds = 2
        self.adapt_steps = 80
        self.n_chains = 2
        return self
