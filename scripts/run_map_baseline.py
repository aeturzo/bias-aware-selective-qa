#!/usr/bin/env python
"""Standalone MAP-baseline run ("is MCMC necessary?") for the ICTAI revision.

Runs ONLY the MAP baseline (same tilted objective as the full method,
greedy optimization instead of sampling) on the same benchmark configuration
as the paper-scale robustness sweep, then inserts/updates its row in
paper_ictai_fable/tables/baselines_body.tex (and mirrors to paper_overleaf).

    python scripts/run_map_baseline.py --fast                 # smoke
    python scripts/run_map_baseline.py --seeds 0,1,2,3,4 --n 1500 \
        --chain-length 5000 --burn-in 1200                    # paper scale
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bias_aware_qa.config import ExperimentConfig
from bias_aware_qa.full_benchmark import build_mixed
from bias_aware_qa.data_trust import DataTrustConfig
from bias_aware_qa.evidence_qa import MockCompassQA
from bias_aware_qa.experiment import score, proportion_cis, cluster_bootstrap_ci
from bias_aware_qa.baselines_extra import MAPBaselineQA
from bias_aware_qa.memory_anchor import seed_memory_from_domains


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--chain-length", type=int, default=5000)
    ap.add_argument("--burn-in", type=int, default=1200)
    ap.add_argument("--adapt-rounds", type=int, default=4)
    ap.add_argument("--adapt-steps", type=int, default=200)
    ap.add_argument("--audit-frac", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "results" / "robustness"))
    ap.add_argument("--tex-out", default=str(ROOT / "paper_ictai_fable" / "tables"))
    args = ap.parse_args()

    cfg = ExperimentConfig(seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
                           n_per_domain=args.n)
    dt = DataTrustConfig(chain_length=args.chain_length, burn_in=args.burn_in,
                         adapt_rounds=args.adapt_rounds, adapt_steps=args.adapt_steps)
    if args.fast:
        cfg.seeds, cfg.n_per_domain = [0], 500
        dt = DataTrustConfig(chain_length=600, burn_in=150, adapt_rounds=2, adapt_steps=80)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tex = Path(args.tex_out)

    bench = build_mixed(cfg)
    mem = seed_memory_from_domains(bench.domains, frac=args.audit_frac, seed=0)
    t0 = time.perf_counter()
    rows = MAPBaselineQA(MockCompassQA(), dt, memory_reference=mem).run(bench)
    s = score(rows)
    s.update({"variant": "map_baseline", "runtime_s": round(time.perf_counter() - t0, 1)})
    print("[map_baseline]", {k: round(v, 3) for k, v in s.items()
                             if isinstance(v, float) and v == v})
    print("[wilson]", proportion_cis(rows))
    print("[cluster MAE CI]", cluster_bootstrap_ci(rows, "mean_abs_error_answered"))

    pd.DataFrame(rows).to_csv(out / "map_baseline_rows.csv", index=False)
    pd.DataFrame([s]).to_csv(out / "map_baseline_score.csv", index=False)

    # insert/update the MAP row in the baselines table body (before the full row)
    def fmt(v, nd=3):
        return "---" if v != v else f"{v:.{nd}f}"
    row = (f"Mode search (same objective, no sampling) & {fmt(s['confidently_wrong_on_biased'])} & "
           f"{fmt(s['interval_coverage_of_clean'])} & {fmt(s['accuracy_answered'])} & "
           f"{fmt(s['mean_abs_error_answered'])} & {fmt(s['coverage'], 2)} \\\\")
    for tdir in [tex, ROOT / "paper_overleaf" / "tables"]:
        body = tdir / "baselines_body.tex"
        if not body.exists():
            continue
        lines = [l for l in body.read_text().splitlines() if l.strip()]
        lines = [l for l in lines if not l.startswith(("MAP (", "Mode search ("))]
        idx = next((i for i, l in enumerate(lines) if l.startswith("Bias-aware")), len(lines))
        lines.insert(idx, row)
        body.write_text("\n".join(lines) + "\n")
        print(f"updated {body}")


if __name__ == "__main__":
    main()
