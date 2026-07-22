#!/usr/bin/env python
"""Fast end-to-end demo of bias-aware selective QA.

Runs a small biased-attribute benchmark, executes naive COMPASS-style QA and the
bias-aware pipeline (real latent-bias MCMC), and prints the comparison table.

    python scripts/run_demo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bias_aware_qa.benchmark import BiasedAttributeBenchmark
from bias_aware_qa.data_trust import DataTrustConfig
from bias_aware_qa.pipeline import NaiveEvidenceQA, BiasAwareQA
from bias_aware_qa.experiment import compare, score


def main() -> None:
    t0 = time.perf_counter()
    print("Building biased-attribute benchmark (synthetic DPP) ...")
    bench = BiasedAttributeBenchmark(n=600, bias_strength=1.3,
                                     missing_eu=0.10, missing_gs=0.40).build(seeds=[0, 1, 2])
    print("  benchmark:", bench.summary())

    print("\nRunning naive COMPASS-style QA (evidence only) ...")
    naive_rows = NaiveEvidenceQA().run(bench)

    print("Running bias-aware QA (evidence + latent-bias MCMC) ...")
    dt = DataTrustConfig(chain_length=1200, burn_in=300, adapt_rounds=3, adapt_steps=150)
    ba_rows = BiasAwareQA(data_trust_config=dt, tau=0.35).run(bench, verbose=True)

    print("\n=== Per-system scores ===")
    for name, rows in [("naive", naive_rows), ("bias_aware", ba_rows)]:
        print(f"[{name}]", {k: round(v, 4) for k, v in score(rows).items()})

    print("\n=== Comparison ===")
    table = compare(naive_rows, ba_rows)
    print(table.to_string(index=False))

    out = ROOT / "results" / "demo_comparison.csv"
    table.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(f"Total demo time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
