# Bias-Aware Selective QA

Artifact accompanying the ICTAI 2026 paper “Beyond Grounding: Bias-Aware
Selective LLM Question Answering over Biased Records.”

This repository contains a self-contained implementation of bias-aware selective
question answering for numeric records with systematic measurement bias. The
core system couples an evidence-grounded QA layer with a constraint-calibrated
latent-bias MCMC layer. It returns a debiased interval-valued answer when the
correction is sufficiently trustworthy, and abstains when uncertainty or anchor
sensitivity is too high.

## Contents

```text
bias_aware_qa/      Python package
scripts/            Experiment and analysis entry points
tests/              Smoke and behavior tests
ontology/           Auditable DPP anchor triples
real_data/          Tiny Climate TRACE sample used only by tests
docs/                Equation derivations and mathematical verification
requirements.txt    Runtime/test dependencies
```

The equation-by-equation supplement, its executable verification script, and
the generated audit are in
[`docs/ictai2026_math_supplement`](docs/ictai2026_math_supplement/).

Large public datasets and generated paper results are intentionally not
included in this repository. The code paths support NHANES, Open Food
Facts, and Climate TRACE inputs when those public files are downloaded locally.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q
python scripts/run_full_experiment.py --smoke
```

The smoke run is API-free and uses the mock evidence layer. It should show the
bias-aware system reducing confidently-wrong answers and improving clean-value
interval coverage relative to the naive evidence-only baseline.

## Main Commands

Paper-scale robustness:

```bash
python scripts/run_robustness.py --seeds 0,1,2,3,4 --n 1500 \
  --chain-length 5000 --burn-in 1200 --adapt-rounds 4 --adapt-steps 200 \
  --n-chains 4 --out results/robustness --tex-out paper_ictai_fable/tables
```

MAP baseline:

```bash
python scripts/run_map_baseline.py --seeds 0,1,2,3,4 --n 1500 \
  --chain-length 5000 --burn-in 1200
```

Main synthetic DPP experiment:

```bash
python scripts/run_full_experiment.py --seeds 0,1,2,3,4 --n 1500 \
  --domains battery,lexmark,viessmann,openfoodfacts \
  --chain-length 10000 --burn-in 2500 --adapt-rounds 4 --adapt-steps 200 \
  --n-chains 4 --seed-memory-frac 0.25 --out results/paper_full_v2
```

## Notes

- No API key is required for the smoke tests or synthetic robustness runs.
- The optional real LLM evidence path reads credentials from environment
  variables; no credentials are included in this repository.
- Generated outputs are ignored by default.
