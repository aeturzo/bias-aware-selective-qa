#!/usr/bin/env python
"""Full ICTAI experiment: all domains x all question types.

Runs the bias-aware numeric layer (real latent-bias MCMC) across every domain
profile, compares against naive COMPASS-style QA, optionally uses the real
COMPASS answerer for evidence and merges COMPASS text-QA (logic/open/recall)
results, and emits every paper table.

    # fast smoke test (no OpenAI needed):
    python scripts/run_full_experiment.py --smoke

    # full run:
    python scripts/run_full_experiment.py --seeds 0,1,2,3,4 --n 1500 \
        --use-compass --compass-csv ../artifacts/eval_joined_XXXX_calibrated.csv

Outputs (results/full/): rows_all.csv, main_comparison.csv, by_domain.csv,
by_type.csv, ablations.csv, mcmc_diagnostics.csv, scores.json, SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bias_aware_qa.config import DOMAIN_PROFILES, CHAIN_SCALE, ExperimentConfig
from bias_aware_qa.full_benchmark import build_mixed
from bias_aware_qa.data_trust import DataTrustConfig, DataTrustLayer
from bias_aware_qa.pipeline import NaiveEvidenceQA, BiasAwareQA
from bias_aware_qa.evidence_qa import MockCompassQA
from bias_aware_qa.experiment import compare, score, score_by
from bias_aware_qa.calibrate import calibrated_report
from bias_aware_qa.env_loader import load_dotenv, openai_available


def _md(df: pd.DataFrame) -> str:
    """Markdown table without requiring the optional `tabulate` package."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def _dt_config(cfg: ExperimentConfig, **over) -> DataTrustConfig:
    base = dict(chain_length=cfg.chain_length, burn_in=cfg.burn_in,
                adapt_rounds=cfg.adapt_rounds, adapt_steps=cfg.adapt_steps,
                n_chains=cfg.n_chains)
    base.update(over)
    return DataTrustConfig(**base)


def _notes(
    *,
    command: str,
    elapsed_s: float,
    cfg: ExperimentConfig,
    evidence_name: str,
    warnings: list[str],
    main_tbl: pd.DataFrame,
    diag: pd.DataFrame,
) -> str:
    lines = [
        "# Results notes",
        "",
        f"Command: `{command}`",
        f"Wall-clock time: {elapsed_s:.1f}s",
        f"GEN_MODEL: `{os.environ.get('GEN_MODEL', '')}`",
        f"OPENAI_EMBEDDING_MODEL: `{os.environ.get('OPENAI_EMBEDDING_MODEL', '')}`",
        f"OPENAI_RESPONSES_DISABLED: `{os.environ.get('OPENAI_RESPONSES_DISABLED', '')}`",
        f"LLM_DISABLED: `{os.environ.get('LLM_DISABLED', '')}`",
        f"OpenAI key present: {openai_available()}",
        f"Evidence layer: {evidence_name}",
        (
            "MCMC: "
            f"chain_length={cfg.chain_length}, burn_in={cfg.burn_in}, "
            f"adapt_rounds={cfg.adapt_rounds}, adapt_steps={cfg.adapt_steps}, "
            f"n_chains={cfg.n_chains}"
        ),
        f"Domains: {cfg.domains}",
        f"Seeds: {cfg.seeds}",
        f"n_per_domain: {cfg.n_per_domain}",
        f"COMPASS CSVs: {cfg.compass_eval_csvs}",
        f"OFF CSV: {cfg.off_csv or ''}",
        "",
        "## Main comparison",
        "",
        _md(main_tbl),
        "",
        "## MCMC diagnostics",
        "",
        _md(diag),
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in warnings)
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--domains", default=",".join(DOMAIN_PROFILES))
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--chain-length", type=int, default=1500)
    ap.add_argument("--burn-in", type=int, default=400)
    ap.add_argument("--adapt-rounds", type=int, default=4)
    ap.add_argument("--adapt-steps", type=int, default=200)
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--memory-jsonl", default=None,
                    help="JSONL of verified clean facts (COMPASS memory format) to anchor on")
    ap.add_argument("--seed-memory-frac", type=float, default=0.0,
                    help="if >0, simulate an audit: derive the memory anchor from this "
                         "fraction of clean records per domain/group/feature")
    ap.add_argument("--real", choices=["nhanes", "off", "climatetrace"], default=None,
                    help="run on a real biased dataset instead of the synthetic DPP mix")
    ap.add_argument("--nhanes-dir", default=None, help="dir with NHANES *.xpt files")
    ap.add_argument("--climatetrace-csv", default=None,
                    help="country self_reported vs climatetrace CSV (else illustrative sample)")
    ap.add_argument("--use-compass", action="store_true", help="use real COMPASS answerer for evidence")
    ap.add_argument("--off-csv", default=None)
    ap.add_argument("--compass-csv", action="append", default=[], help="COMPASS text-QA eval CSV(s) to merge")
    ap.add_argument("--out", default=str(ROOT / "results" / "full"))
    args = ap.parse_args()

    load_dotenv()
    t0 = time.perf_counter()
    cfg = ExperimentConfig(
        domains=[d.strip() for d in args.domains.split(",") if d.strip()],
        seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
        n_per_domain=args.n, tau=args.tau, use_compass=args.use_compass,
        chain_length=args.chain_length, burn_in=args.burn_in,
        adapt_rounds=args.adapt_rounds, adapt_steps=args.adapt_steps,
        n_chains=args.n_chains,
        off_csv=args.off_csv, compass_eval_csvs=args.compass_csv,
    )
    if args.smoke:
        cfg.smoke()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if cfg.off_csv and args.real != "off":
        warnings.append(
            "--off-csv is recorded for reproducibility, but the current numeric "
            "bias-aware benchmark still uses the configured Open Food Facts "
            "synthetic profile; real OFF nutrition columns are not wired into "
            "the data-trust query generator."
        )

    if args.real:
        from bias_aware_qa.benchmark import BiasedAttributeBenchmark
        from bias_aware_qa import real_data
        bench = BiasedAttributeBenchmark(n=cfg.n_per_domain)
        if args.real == "nhanes":
            if not args.nhanes_dir:
                raise SystemExit("--real nhanes requires --nhanes-dir <path to *.xpt>")
            bench.domains = real_data.nhanes_domains(args.nhanes_dir, seeds=cfg.seeds, max_rows=cfg.n_per_domain)
        elif args.real == "off":
            if not cfg.off_csv:
                raise SystemExit("--real off requires --off-csv <path to OFF export>")
            bench.domains = real_data.openfoodfacts_domains(cfg.off_csv, seeds=cfg.seeds, max_rows=cfg.n_per_domain)
        elif args.real == "climatetrace":
            bench.domains = real_data.climatetrace_domains(args.climatetrace_csv, seeds=cfg.seeds)
        print(f"REAL dataset: {args.real} | seeds: {cfg.seeds}")
    else:
        print(f"Domains: {cfg.domains} | seeds: {cfg.seeds} | n/domain: {cfg.n_per_domain}")
        bench = build_mixed(cfg)
    print("benchmark:", bench.summary())

    # Evidence layer: real COMPASS answerer or mock.
    evidence = MockCompassQA()
    if cfg.use_compass:
        from bias_aware_qa.compass_evidence import CompassEvidenceQA
        evidence = CompassEvidenceQA(enable_llm=True)
        print(f"[evidence] COMPASS LLM ready = {evidence.llm_ready} (OpenAI key present = {openai_available()})")
        if not evidence.llm_ready:
            warnings.append("--use-compass was requested, but COMPASS/OpenAI was unavailable; evidence fell back to mock.")

    # External-memory clean-reference anchor (optional, strongest anchor).
    memory_ref = None
    if args.memory_jsonl:
        from bias_aware_qa.memory_anchor import MemoryCleanReference
        memory_ref = MemoryCleanReference.from_jsonl(args.memory_jsonl)
        print(f"[memory] loaded {memory_ref.n_facts} verified clean facts from {args.memory_jsonl}")
    elif args.seed_memory_frac > 0:
        from bias_aware_qa.memory_anchor import seed_memory_from_domains
        memory_ref = seed_memory_from_domains(
            bench.domains, frac=args.seed_memory_frac, seed=0,
            out_path=out / "verified_clean_memory.jsonl")
        print(f"[memory] seeded {memory_ref.n_facts} verified clean facts (sim-audit "
              f"frac={args.seed_memory_frac}) -> {out / 'verified_clean_memory.jsonl'}")

    dt = _dt_config(cfg)
    print("Running naive + bias-aware over the full mix ...")
    naive_rows = NaiveEvidenceQA(evidence).run(bench)
    ba = BiasAwareQA(evidence, dt, tau=cfg.tau, chain_scale_by_domain=CHAIN_SCALE,
                     memory_reference=memory_ref)
    ba_rows = ba.run(bench, verbose=True)

    # Merge real COMPASS text-QA results (logic/open/recall) if provided.
    compass_rows: list[dict] = []
    if cfg.compass_eval_csvs:
        from bias_aware_qa.compass_evidence import load_compass_text_results
        compass_rows = load_compass_text_results(cfg.compass_eval_csvs)
        print(f"[merge] loaded {len(compass_rows)} COMPASS text-QA records")

    # ---- Tables ----
    all_rows = ([{**r, "system": "naive"} for r in naive_rows]
                + [{**r, "system": "bias_aware"} for r in ba_rows]
                + compass_rows)
    pd.DataFrame(all_rows).to_csv(out / "rows_all.csv", index=False)

    main_tbl = compare(naive_rows, ba_rows)
    main_tbl.to_csv(out / "main_comparison.csv", index=False)

    by_dom = score_by(ba_rows, "domain_profile").rename(columns={"domain_profile": "domain"})
    by_dom.to_csv(out / "by_domain.csv", index=False)
    # unified type breakdown: numeric types + (merged) text types
    type_rows = ba_rows + compass_rows if compass_rows else ba_rows
    by_type = score_by(type_rows, "qtype"); by_type.to_csv(out / "by_type.csv", index=False)

    # ---- Ablations ----
    print("Running ablations (constraint off; sensitivity off) ...")
    ab_noconstraint = BiasAwareQA(evidence, _dt_config(cfg, constrained=False), tau=cfg.tau,
                                  chain_scale_by_domain=CHAIN_SCALE, memory_reference=memory_ref).run(bench)
    ab_nosens = BiasAwareQA(evidence, dt, tau=cfg.tau, use_sensitivity=False,
                            chain_scale_by_domain=CHAIN_SCALE, memory_reference=memory_ref).run(bench)
    ablation = pd.DataFrame([
        {"variant": "full", **score(ba_rows)},
        {"variant": "no_constraint", **score(ab_noconstraint)},
        {"variant": "no_sensitivity", **score(ab_nosens)},
        {"variant": "naive", **score(naive_rows)},
    ])
    ablation.to_csv(out / "ablations.csv", index=False)

    diag = pd.DataFrame(ba.diagnostics_)
    diag.to_csv(out / "mcmc_diagnostics.csv", index=False)

    # ---- Post-hoc calibration (fit on dev seeds, evaluate ECE on held-out test) ----
    seeds = list(cfg.seeds)
    dev_seeds, test_seeds = (seeds[:-1], seeds[-1:]) if len(seeds) >= 2 else (seeds, seeds)
    calib = calibrated_report(ba_rows, dev_seeds, test_seeds)
    calib.update({"dev_seeds": dev_seeds, "test_seeds": test_seeds})
    pd.DataFrame([calib]).to_csv(out / "calibration.csv", index=False)
    print("[calibration]", calib)
    # Leave-one-seed-out cross-validated calibration (mean +/- sd over folds):
    # far less fragile than the single tiny held-out split above.
    from bias_aware_qa.calibrate import loso_calibration_report
    loso = loso_calibration_report(ba_rows, seeds)
    if loso.get("n_folds"):
        pd.DataFrame(loso["folds"]).to_csv(out / "calibration_loso_folds.csv", index=False)
        (out / "calibration_loso.json").write_text(json.dumps(
            {k: v for k, v in loso.items() if k != "folds"}, indent=2))
        print("[calibration LOSO]", {k: v for k, v in loso.items() if k != "folds"})

    anchor_source = getattr(ba, "anchor_source_", None)
    print(f"[anchor] bias-correction anchor source = {anchor_source}")
    (out / "scores.json").write_text(json.dumps(
        {"naive": score(naive_rows), "bias_aware": score(ba_rows), "calibration": calib,
         "anchor_source": anchor_source,
         "benchmark": bench.summary(), "domains": cfg.domains, "seeds": cfg.seeds}, indent=2))

    # ---- Markdown summary ----
    lines = ["# Full experiment summary\n",
             f"Domains: {', '.join(cfg.domains)} | seeds: {cfg.seeds} | n/domain: {cfg.n_per_domain}",
             f"Benchmark: {bench.summary()}",
             f"Evidence layer: {'real COMPASS' if cfg.use_compass else 'mock'}\n",
             "## Main comparison\n", _md(main_tbl),
             "\n## By domain (bias-aware)\n", _md(by_dom.round(4)),
             "\n## By type\n", _md(by_type.round(4)),
             "\n## Ablations\n", _md(ablation.round(4)),
             "\n## Calibration (held-out test split)\n",
             f"Raw ECE {calib['ece_raw_test']} -> calibrated ECE {calib['ece_calibrated_test']} "
             f"(test n={calib['n_test']}, accuracy {calib['test_accuracy']}, "
             f"mean confidence {calib['mean_conf_raw_test']} -> {calib['mean_conf_cal_test']}).",
             "\n## MCMC diagnostics\n", _md(diag)]
    (out / "SUMMARY.md").write_text("\n".join(lines))
    elapsed_s = time.perf_counter() - t0
    evidence_name = "real COMPASS" if cfg.use_compass else "mock"
    (out / "RESULTS_NOTES.md").write_text(_notes(
        command="python " + " ".join(sys.argv),
        elapsed_s=elapsed_s,
        cfg=cfg,
        evidence_name=evidence_name,
        warnings=warnings,
        main_tbl=main_tbl,
        diag=diag,
    ))

    print("\n=== MAIN COMPARISON ===")
    print(main_tbl.to_string(index=False))
    print(f"\nWrote all tables to {out}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    print(f"Total time: {elapsed_s:.1f}s")


if __name__ == "__main__":
    main()
