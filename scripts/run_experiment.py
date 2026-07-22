#!/usr/bin/env python
"""Full experiment for the ICTAI paper (larger, multi-seed).

    python scripts/run_experiment.py --seeds 0,1,2,3,4 --n 1500 \
        --chain-length 2500 --burn-in 600 --out results

Writes: results/rows_naive.csv, results/rows_bias_aware.csv,
        results/comparison.csv, results/scores.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bias_aware_qa.benchmark import BiasedAttributeBenchmark
from bias_aware_qa.data_trust import DataTrustConfig
from bias_aware_qa.pipeline import NaiveEvidenceQA, BiasAwareQA
from bias_aware_qa.experiment import compare, score


def _seeds(raw: str) -> list[int]:
    return [int(s) for s in raw.split(",") if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--bias-strength", type=float, default=1.3)
    ap.add_argument("--missing-gs", type=float, default=0.40)
    ap.add_argument("--chain-length", type=int, default=2500)
    ap.add_argument("--burn-in", type=int, default=600)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bench = BiasedAttributeBenchmark(
        n=args.n, bias_strength=args.bias_strength, missing_gs=args.missing_gs
    ).build(seeds=_seeds(args.seeds))
    print("benchmark:", bench.summary())

    naive_rows = NaiveEvidenceQA().run(bench)
    dt = DataTrustConfig(chain_length=args.chain_length, burn_in=args.burn_in,
                         adapt_rounds=4, adapt_steps=200)
    ba_rows = BiasAwareQA(data_trust_config=dt, tau=args.tau).run(bench, verbose=True)

    pd.DataFrame(naive_rows).to_csv(out / "rows_naive.csv", index=False)
    pd.DataFrame(ba_rows).to_csv(out / "rows_bias_aware.csv", index=False)
    table = compare(naive_rows, ba_rows)
    table.to_csv(out / "comparison.csv", index=False)
    (out / "scores.json").write_text(json.dumps(
        {"naive": score(naive_rows), "bias_aware": score(ba_rows),
         "benchmark": bench.summary()}, indent=2))

    print("\n=== Comparison ===")
    print(table.to_string(index=False))
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
