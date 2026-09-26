#!/usr/bin/env python
"""Anchor-robustness experiments for the ICTAI revision.

Answers the reviewer question "is the method just reproducing a supplied
target?" with four experiment families on the controlled DPP benchmark
(numeric questions, mock evidence; no API needed):

  E1  Baselines:      anchor-only (no MCMC) and matched-coverage selective RAG.
  E2  Audit size:     memory anchor from a simulated audit of {1,5,10,25}% of
                      clean records; plus a *disjoint* audit whose facts come
                      from independently generated records of the same
                      population (audit seeds != eval seeds).
  E3  Anchor perturbation: resolved anchor scaled by {0.5,0.75,0.9,1.1,1.25,1.5}
                      and a wrong-sign anchor -- measure how error and
                      abstention respond.
  E4  Ontology-only:  weak prior anchor (no audit facts at all).

Outputs: CSVs under --out and LaTeX table fragments under --tex-out.

    python scripts/run_robustness.py --fast              # reduced (sandbox)
    python scripts/run_robustness.py --seeds 0,1,2,3,4 --n 1500 \
        --chain-length 2500 --burn-in 600 --n-chains 4   # paper scale
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bias_aware_qa.config import DOMAIN_PROFILES, CHAIN_SCALE, ExperimentConfig
from bias_aware_qa.full_benchmark import build_mixed
from bias_aware_qa.data_trust import DataTrustConfig
from bias_aware_qa.pipeline import NaiveEvidenceQA, BiasAwareQA
from bias_aware_qa.evidence_qa import MockCompassQA
from bias_aware_qa.experiment import score, bootstrap_table
from bias_aware_qa.baselines_extra import AnchorOnlyQA, MAPBaselineQA, apply_matched_coverage
from bias_aware_qa.memory_anchor import seed_memory_from_domains

AUDIT_FRACS = [0.01, 0.05, 0.10, 0.25]
ANCHOR_SCALES = [0.5, 0.75, 0.9, 1.1, 1.25, 1.5]
KEY_METRICS = ["confidently_wrong_on_biased", "interval_coverage_of_clean",
               "accuracy_answered", "mean_abs_error_answered", "coverage",
               "mean_interval_width_answered"]


def _fmt(v: float, nd: int = 3) -> str:
    if v != v:
        return "---"
    return f"{v:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="reduced settings (smoke-scale)")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--chain-length", type=int, default=2500)
    ap.add_argument("--burn-in", type=int, default=600)
    ap.add_argument("--adapt-rounds", type=int, default=4)
    ap.add_argument("--adapt-steps", type=int, default=200)
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--audit-frac-main", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "results" / "robustness"))
    ap.add_argument("--tex-out", default=str(ROOT / "paper_ictai_fable" / "tables"))
    args = ap.parse_args()

    cfg = ExperimentConfig(
        seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
        n_per_domain=args.n, tau=args.tau,
        chain_length=args.chain_length, burn_in=args.burn_in,
        adapt_rounds=args.adapt_rounds, adapt_steps=args.adapt_steps,
        n_chains=args.n_chains)
    if args.fast:
        cfg.seeds = [0]
        cfg.n_per_domain = 500
        cfg.chain_length, cfg.burn_in = 600, 150
        cfg.adapt_rounds, cfg.adapt_steps = 2, 80
        cfg.n_chains = 2
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tex = Path(args.tex_out); tex.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def dt(**over) -> DataTrustConfig:
        base = dict(chain_length=cfg.chain_length, burn_in=cfg.burn_in,
                    adapt_rounds=cfg.adapt_rounds, adapt_steps=cfg.adapt_steps,
                    n_chains=cfg.n_chains)
        base.update(over)
        return DataTrustConfig(**base)

    print(f"seeds={cfg.seeds} n={cfg.n_per_domain} chains={cfg.n_chains}x{cfg.chain_length}")
    bench = build_mixed(cfg)
    print("benchmark:", bench.summary())
    evidence = MockCompassQA()

    # main-audit memory anchor (same-benchmark audit, as in the paper run)
    mem_main = seed_memory_from_domains(bench.domains, frac=args.audit_frac_main, seed=0,
                                        out_path=out / "audit_main.jsonl")
    # DISJOINT audit: facts from independently generated domain instances
    # (same profiles/population, audit seeds offset by +1000 => zero record overlap)
    cfg_dis = ExperimentConfig(**{**cfg.__dict__,
                                  "seeds": [s + 1000 for s in cfg.seeds]})
    bench_dis = build_mixed(cfg_dis)
    mem_disjoint = seed_memory_from_domains(bench_dis.domains, frac=args.audit_frac_main,
                                            seed=1, out_path=out / "audit_disjoint.jsonl")

    def run_ba(tag: str, memory_ref=None, **dtover) -> dict:
        t = time.perf_counter()
        rows = BiasAwareQA(evidence, dt(**dtover), tau=cfg.tau,
                           chain_scale_by_domain=CHAIN_SCALE,
                           memory_reference=memory_ref).run(bench)
        s = score(rows)
        s.update({"variant": tag, "runtime_s": round(time.perf_counter() - t, 1)})
        if s.get("n_queries"):
            s["runtime_s_per_query"] = round(s["runtime_s"] / s["n_queries"], 3)
        print(f"[{tag}] " + " ".join(f"{k}={_fmt(s.get(k))}" for k in KEY_METRICS)
              + f" ({s['runtime_s']}s)")
        return {"rows": rows, "score": s}

    results: list[dict] = []
    rows_store: dict[str, list[dict]] = {}

    # ---- E1: baselines --------------------------------------------------
    naive_rows = NaiveEvidenceQA(evidence).run(bench)
    s = score(naive_rows); s["variant"] = "naive"; results.append(s)
    rows_store["naive"] = naive_rows

    full = run_ba("full_memory_anchor", memory_ref=mem_main)
    results.append(full["score"]); rows_store["full_memory_anchor"] = full["rows"]
    target_cov = full["score"]["coverage"]

    matched = apply_matched_coverage(naive_rows, target_cov)
    s = score(matched); s["variant"] = f"naive_matched_cov_{target_cov:.2f}"
    results.append(s); rows_store["naive_matched_cov"] = matched
    print(f"[naive_matched_cov] " + " ".join(f"{k}={_fmt(s.get(k))}" for k in KEY_METRICS))

    anchor_only = AnchorOnlyQA(evidence, dt(), memory_reference=mem_main).run(bench)
    s = score(anchor_only); s["variant"] = "anchor_only"
    results.append(s); rows_store["anchor_only"] = anchor_only
    print(f"[anchor_only] " + " ".join(f"{k}={_fmt(s.get(k))}" for k in KEY_METRICS))

    anchor_only_ont = AnchorOnlyQA(evidence, dt()).run(bench)   # ontology, no audit
    s = score(anchor_only_ont); s["variant"] = "anchor_only_ontology"
    results.append(s); rows_store["anchor_only_ontology"] = anchor_only_ont

    # MAP baseline: same tilted objective, optimization instead of sampling
    t = time.perf_counter()
    map_rows = MAPBaselineQA(evidence, dt(), memory_reference=mem_main).run(bench)
    s = score(map_rows); s["variant"] = "map_baseline"
    s["runtime_s"] = round(time.perf_counter() - t, 1)
    results.append(s); rows_store["map_baseline"] = map_rows
    print(f"[map_baseline] " + " ".join(f"{k}={_fmt(s.get(k))}" for k in KEY_METRICS))

    # ---- E2: audit size + disjoint audit --------------------------------
    for frac in AUDIT_FRACS:
        mem = (mem_main if frac == args.audit_frac_main else
               seed_memory_from_domains(bench.domains, frac=frac, seed=0))
        r = run_ba(f"audit_frac_{frac:g}", memory_ref=mem)
        r["score"]["audit_frac"] = frac
        results.append(r["score"])
        # anchor-only at the same audit size (does MCMC add anything?)
        ao = AnchorOnlyQA(evidence, dt(), memory_reference=mem).run(bench)
        s = score(ao); s["variant"] = f"anchor_only_frac_{frac:g}"; s["audit_frac"] = frac
        results.append(s)

    r = run_ba("audit_disjoint", memory_ref=mem_disjoint)
    results.append(r["score"]); rows_store["audit_disjoint"] = r["rows"]

    # ---- E4: ontology-only (weak prior, no audit) ------------------------
    r = run_ba("ontology_anchor", memory_ref=None)
    results.append(r["score"]); rows_store["ontology_anchor"] = r["rows"]

    # ---- E3: anchor perturbation -----------------------------------------
    for sc in ANCHOR_SCALES:
        r = run_ba(f"anchor_scale_{sc:g}", memory_ref=mem_main, anchor_scale=sc)
        r["score"]["anchor_scale"] = sc
        results.append(r["score"])
    r = run_ba("anchor_wrong_sign", memory_ref=mem_main, anchor_flip_sign=True)
    r["score"]["anchor_scale"] = -1.0
    results.append(r["score"])

    # ---- outputs ---------------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv(out / "robustness.csv", index=False)

    boot = bootstrap_table({k: v for k, v in rows_store.items()},
                           KEY_METRICS, n_boot=1000)
    boot.to_csv(out / "bootstrap_cis.csv", index=False)

    # LaTeX fragments -------------------------------------------------------
    def row_of(tag_prefix: str) -> dict | None:
        hits = [r for r in results if str(r.get("variant", "")).startswith(tag_prefix)]
        return hits[0] if hits else None

    # baselines table body
    lines = []
    label = {"naive": "Naive RAG/LLM",
             "naive_matched_cov": "Selective RAG (matched coverage)",
             "anchor_only": "Anchor-only (no MCMC)",
             "map_baseline": "Mode search (same objective, no sampling)",
             "full_memory_anchor": "Bias-aware (full)"}
    for tag in ["naive", "naive_matched_cov", "anchor_only", "map_baseline", "full_memory_anchor"]:
        r = row_of(tag)
        if r:
            lines.append(
                f"{label[tag]} & {_fmt(r['confidently_wrong_on_biased'])} & "
                f"{_fmt(r['interval_coverage_of_clean'])} & {_fmt(r['accuracy_answered'])} & "
                f"{_fmt(r['mean_abs_error_answered'])} & {_fmt(r['coverage'], 2)} \\\\")
    (tex / "baselines_body.tex").write_text("\n".join(lines) + "\n")

    # audit-size table body
    lines = []
    for frac in AUDIT_FRACS:
        r = row_of(f"audit_frac_{frac:g}")
        a = row_of(f"anchor_only_frac_{frac:g}")
        if r:
            lines.append(
                f"{int(frac*100)}\\% & {_fmt(r['mean_abs_error_answered'])} & "
                f"{_fmt(r['interval_coverage_of_clean'])} & {_fmt(r['coverage'], 2)} & "
                f"{_fmt(a['mean_abs_error_answered']) if a else '---'} \\\\")
    r = row_of("audit_disjoint")
    if r:
        lines.append(
            f"25\\% (disjoint records) & {_fmt(r['mean_abs_error_answered'])} & "
            f"{_fmt(r['interval_coverage_of_clean'])} & {_fmt(r['coverage'], 2)} & --- \\\\")
    r = row_of("ontology_anchor")
    if r:
        lines.append(
            f"none (ontology prior only) & {_fmt(r['mean_abs_error_answered'])} & "
            f"{_fmt(r['interval_coverage_of_clean'])} & {_fmt(r['coverage'], 2)} & --- \\\\")
    (tex / "audit_size_body.tex").write_text("\n".join(lines) + "\n")

    # perturbation table body
    lines = []
    for sc in ANCHOR_SCALES + [-1.0]:
        tag = "anchor_wrong_sign" if sc == -1.0 else f"anchor_scale_{sc:g}"
        name = "wrong sign" if sc == -1.0 else f"$\\times{sc:g}$"
        r = row_of(tag)
        if r:
            lines.append(
                f"{name} & {_fmt(r['mean_abs_error_answered'])} & "
                f"{_fmt(r['accuracy_answered'])} & {_fmt(r['coverage'], 2)} & "
                f"{_fmt(r['confidently_wrong_on_biased'])} \\\\")
    (tex / "perturbation_body.tex").write_text("\n".join(lines) + "\n")

    meta = {"config": {"seeds": cfg.seeds, "n_per_domain": cfg.n_per_domain,
                       "chain_length": cfg.chain_length, "burn_in": cfg.burn_in,
                       "n_chains": cfg.n_chains, "fast": bool(args.fast)},
            "elapsed_s": round(time.perf_counter() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out}/robustness.csv and LaTeX bodies to {tex}/ "
          f"({meta['elapsed_s']}s total)")


if __name__ == "__main__":
    main()
