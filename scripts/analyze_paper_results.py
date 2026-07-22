#!/usr/bin/env python
"""Post-hoc analyses of the existing paper run (results/paper_full).

Computes, at full paper scale and WITHOUT rerunning any MCMC:
  * question-level bootstrap 95% CIs for the main-table metrics;
  * leave-one-seed-out calibrated ECE / Brier (replaces the fragile 12-question
    held-out split);
  * matched-coverage selective-RAG baseline (naive rows re-thresholded to the
    bias-aware system's coverage, using the naive system's own confidences);
  * anchor-only baseline at paper scale (deterministic benchmark rebuild + the
    saved audit memory; anchor resolution identical to the paper run);
  * mean interval widths.

    python scripts/analyze_paper_results.py --run results/paper_full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bias_aware_qa.config import ExperimentConfig
from bias_aware_qa.full_benchmark import build_mixed
from bias_aware_qa.data_trust import DataTrustConfig
from bias_aware_qa.experiment import score, bootstrap_table
from bias_aware_qa.calibrate import loso_calibration_report
from bias_aware_qa.baselines_extra import AnchorOnlyQA, apply_matched_coverage
from bias_aware_qa.memory_anchor import MemoryCleanReference

METRICS = ["confidently_wrong_on_biased", "interval_coverage_of_clean",
           "accuracy_answered", "mean_abs_error_answered", "coverage",
           "aurc_biased_subset", "ece", "mean_interval_width_answered"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "results" / "paper_full"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    run = Path(args.run)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows_all = pd.read_csv(run / "rows_all.csv")
    naive = rows_all[rows_all["system"] == "naive"].to_dict("records")
    ba = rows_all[rows_all["system"] == "bias_aware"].to_dict("records")
    # numeric-only subsets (the text rows have no clean values)
    naive = [r for r in naive if str(r.get("kind")) in ("gap", "mean")]
    ba = [r for r in ba if str(r.get("kind")) in ("gap", "mean")]
    print(f"loaded {len(naive)} naive + {len(ba)} bias-aware numeric rows")

    out: dict = {}

    # ---- matched-coverage selective RAG ----------------------------------
    target_cov = score(ba)["coverage"]
    matched = apply_matched_coverage(naive, target_cov)
    out["naive_matched_cov"] = score(matched)
    print("[matched-cov naive]", json.dumps(out["naive_matched_cov"], indent=None, default=float))

    # ---- anchor-only at paper scale ---------------------------------------
    cfg = ExperimentConfig(seeds=seeds, n_per_domain=args.n)
    bench = build_mixed(cfg)
    mem = MemoryCleanReference.from_jsonl(run / "verified_clean_memory.jsonl")
    print(f"[anchor-only] rebuilt benchmark {bench.summary()}, {mem.n_facts} audit facts")
    anchor_rows = AnchorOnlyQA(data_trust_config=DataTrustConfig(),
                               memory_reference=mem).run(bench)
    out["anchor_only"] = score(anchor_rows)
    print("[anchor-only]", json.dumps(out["anchor_only"], indent=None, default=float))

    # ---- bootstrap CIs -----------------------------------------------------
    boot = bootstrap_table(
        {"naive": naive, "bias_aware": ba,
         "naive_matched_cov": matched, "anchor_only": anchor_rows},
        METRICS, n_boot=args.n_boot)
    boot.to_csv(run / "bootstrap_cis.csv", index=False)
    print(boot.to_string(index=False))

    # ---- LOSO calibration --------------------------------------------------
    loso = loso_calibration_report(ba, seeds)
    out["loso_calibration"] = {k: v for k, v in loso.items() if k != "folds"}
    print("[loso]", json.dumps(out["loso_calibration"], default=float))

    # combined (numeric + text) calibration, the harder/more honest set
    text = rows_all[rows_all["system"] == "compass_text"].to_dict("records")
    if text:
        loso_mix = loso_calibration_report(ba + text, seeds)
        # text rows have no seed column values from the benchmark; guard:
        out["loso_calibration_note"] = "text rows lack benchmark seeds; mixed LOSO uses numeric folds only"

    out["scores"] = {"naive": score(naive), "bias_aware": score(ba)}
    (run / "extra_analyses.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWrote {run}/extra_analyses.json and {run}/bootstrap_cis.csv")


if __name__ == "__main__":
    main()
