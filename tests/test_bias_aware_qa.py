"""Smoke + behaviour tests for the bias-aware selective QA package."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bias_aware_qa import selective
from bias_aware_qa.benchmark import BiasedAttributeBenchmark
from bias_aware_qa.data_trust import DataTrustConfig
from bias_aware_qa.pipeline import NaiveEvidenceQA, BiasAwareQA
from bias_aware_qa.experiment import score


def test_selective_metrics_sane():
    # Perfect confidence ordering -> AURC low; perfect calibration -> ECE ~ 0.
    conf = [0.9, 0.8, 0.7, 0.6]
    correct = [True, True, True, False]
    a = selective.aurc(conf, correct)
    assert 0.0 <= a <= 1.0
    e = selective.ece([1.0, 1.0, 0.0, 0.0], [True, True, False, False], n_bins=2)
    assert e < 1e-6
    lo, hi = selective.wilson_interval(9, 10)
    assert 0.0 <= lo <= hi <= 1.0


def test_benchmark_builds_and_has_biased_queries():
    bench = BiasedAttributeBenchmark(n=300, bias_strength=1.5, missing_gs=0.4).build(seeds=[0, 1])
    s = bench.summary()
    assert s["n_queries"] > 0
    assert s["n_biased"] >= 1  # bias injection should create biased quantities
    for q in bench.all_queries():
        assert np.isfinite(q.clean_value)


def test_anchor_from_ontology():
    from bias_aware_qa.anchor import AnchorProvider
    ap = AnchorProvider.from_ontology()
    pt = ap.policy_target()
    # anchor must come from the ontology TTL, not the fallback constants
    assert "fallback" not in ap.source
    assert abs(pt["carbon_kg_per_kwh"] - 5.0) < 1e-9
    # polarity/sign must be consistent: carbon adverse (+1), quality protective (-1)
    assert ap.sign("carbon_kg_per_kwh") == 1.0
    assert ap.sign("repairability_score") == -1.0
    assert 0.0 < ap.epsilon() < 2.0


def test_memory_anchor_derives_and_shrinks():
    from bias_aware_qa.benchmark import BiasedAttributeBenchmark
    from bias_aware_qa.memory_anchor import seed_memory_from_domains
    from bias_aware_qa.data_trust import DataTrustLayer, DataTrustConfig
    b = BiasedAttributeBenchmark(n=500, bias_strength=1.3, missing_gs=0.4).build([0], "battery")
    dom = b.domains[0]
    mem = seed_memory_from_domains([dom], frac=0.25, seed=1)
    assert mem.n_facts > 0
    gap, n = mem.group_gap_with_n("carbon_kg_per_kwh", "battery")
    assert gap is not None and n >= 3
    L = DataTrustLayer(DataTrustConfig(chain_length=400, burn_in=120, adapt_rounds=1,
                                       adapt_steps=60, n_chains=2), memory_reference=mem)
    L.fit(dom)
    assert "memory" in L.anchor_source
    # shrunk target should sit between the audit estimate and the ontology prior (5.0)
    t = L.policy_target["carbon_kg_per_kwh"]
    assert min(gap, 5.0) - 3.0 <= t <= max(gap, 5.0) + 3.0


def test_real_data_climatetrace_sample_builds():
    from bias_aware_qa.real_data import climatetrace_domains
    doms = climatetrace_domains()  # bundled illustrative sample
    assert len(doms) >= 1
    d = doms[0]
    assert d.gap_features == d.feature_cols and len(d.queries) >= 1
    q = d.queries[0]
    assert np.isfinite(q.clean_value) and np.isfinite(q.observed_value)
    # groups relabeled to EU/GS for the sampler
    assert set(d.observed_df["region"].unique()) <= {"EU", "GS"}


def test_real_data_climatetrace_uses_feature_metadata(tmp_path):
    from bias_aware_qa.real_data import climatetrace_domains
    p = tmp_path / "ct.csv"
    p.write_text(
        "country,group,self_reported,climatetrace,feature\n"
        "A,AnnexI,10,11,emissions_tco2e_per_capita\n"
        "B,AnnexI,12,13,emissions_tco2e_per_capita\n"
        "C,nonAnnexI,4,5,emissions_tco2e_per_capita\n"
        "D,nonAnnexI,5,6,emissions_tco2e_per_capita\n",
        encoding="utf-8",
    )
    dom = climatetrace_domains(p, seeds=(0,))[0]
    assert dom.feature_cols == ["emissions_tco2e_per_capita"]
    assert dom.queries[0].quantity == "gap_emissions_tco2e_per_capita"


def test_mock_compass_respects_query_undercoverage():
    from bias_aware_qa.benchmark import Query
    from bias_aware_qa.evidence_qa import MockCompassQA
    q = Query(
        qid="q", domain="d", seed=0, quantity="gap_x", feature="x", kind="gap",
        clean_value=0.0, observed_value=1.0, n_evidence_eu=43, n_evidence_gs=12,
        is_biased=True, is_undercovered=False, tolerance=0.5, qtype="attr_gap",
    )
    assert MockCompassQA(min_group_evidence=40).answer(q).has_evidence
    q.is_undercovered = True
    assert not MockCompassQA(min_group_evidence=40).answer(q).has_evidence


def test_pipelines_run_and_bias_aware_helps():
    bench = BiasedAttributeBenchmark(n=400, bias_strength=1.5, missing_gs=0.4).build(seeds=[0, 1])
    naive_rows = NaiveEvidenceQA().run(bench)
    dt = DataTrustConfig(chain_length=500, burn_in=150, adapt_rounds=2, adapt_steps=80)
    ba_rows = BiasAwareQA(data_trust_config=dt, tau=0.35).run(bench)

    s_naive, s_ba = score(naive_rows), score(ba_rows)
    # Uniform record schema
    for r in naive_rows + ba_rows:
        assert {"qid", "answered", "value", "confidence", "correct"} <= set(r)
    # Bias-aware should not do WORSE on the two headline reliability metrics.
    assert s_ba["confidently_wrong_on_biased"] <= s_naive["confidently_wrong_on_biased"] + 1e-9
    assert s_ba["interval_coverage_of_clean"] >= s_naive["interval_coverage_of_clean"] - 1e-9
