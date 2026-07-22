"""Scoring and comparison harness for bias-aware vs naive QA."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import selective


def score(rows: list[dict[str, Any]]) -> dict[str, float]:
    df = pd.DataFrame(rows)
    n = len(df)
    ans = df[df["answered"]]
    biased = df[df["is_biased"]]
    biased_ans = biased[biased["answered"]]
    numeric_ans = ans[
        np.isfinite(pd.to_numeric(ans["value"], errors="coerce"))
        & np.isfinite(pd.to_numeric(ans["clean_value"], errors="coerce"))
    ] if len(ans) else ans
    interval_ans = ans[
        np.isfinite(pd.to_numeric(ans["ci_lower"], errors="coerce"))
        & np.isfinite(pd.to_numeric(ans["ci_upper"], errors="coerce"))
        & np.isfinite(pd.to_numeric(ans["clean_value"], errors="coerce"))
    ] if len(ans) else ans

    # Confidently-wrong on biased: answered a biased query with conf>=0.5 and wrong.
    conf_wrong = biased_ans[(biased_ans["confidence"] >= 0.5) & (~biased_ans["correct"])]
    conf_wrong_rate = float(len(conf_wrong) / len(biased)) if len(biased) else float("nan")

    # AURC on the biased subset (rank by confidence).
    if len(biased_ans) >= 2:
        aurc_biased = selective.aurc(biased_ans["confidence"].to_numpy(),
                                     biased_ans["correct"].to_numpy())
    else:
        aurc_biased = float("nan")

    # ECE over answered items.
    ece = selective.ece(ans["confidence"].to_numpy(), ans["correct"].to_numpy()) if len(ans) else float("nan")

    return {
        "n_queries": n,
        "coverage": float(len(ans) / n) if n else float("nan"),
        "accuracy_answered": float(ans["correct"].mean()) if len(ans) else float("nan"),
        "confidently_wrong_on_biased": conf_wrong_rate,
        "interval_coverage_of_clean": float(interval_ans["ci_covers_clean"].mean())
        if len(interval_ans) else float("nan"),
        "ece": ece,
        "brier_answered": float(
            np.mean((ans["confidence"].to_numpy(dtype=float)
                     - ans["correct"].to_numpy(dtype=float)) ** 2)
        ) if len(ans) else float("nan"),
        "aurc_biased_subset": aurc_biased,
        "mean_abs_error_answered": float(
            (pd.to_numeric(numeric_ans["value"], errors="coerce")
             - pd.to_numeric(numeric_ans["clean_value"], errors="coerce")).abs().mean()
        ) if len(numeric_ans) else float("nan"),
        "mean_interval_width_answered": float(
            (pd.to_numeric(interval_ans["ci_upper"], errors="coerce")
             - pd.to_numeric(interval_ans["ci_lower"], errors="coerce")).mean()
        ) if len(interval_ans) else float("nan"),
        "nmae_answered": _nmae(numeric_ans),
        "n_biased": int(len(biased)),
    }


def _nmae(numeric_ans: pd.DataFrame) -> float:
    """Tolerance-normalized MAE on answered numeric questions.

    Raw MAE averages errors across attributes with incompatible units; NMAE
    divides each error by its query tolerance, so 1.0 = 'exactly at the
    correctness boundary' for every attribute and dataset alike.
    """
    if "tolerance" not in numeric_ans.columns or not len(numeric_ans):
        return float("nan")
    tol = pd.to_numeric(numeric_ans["tolerance"], errors="coerce")
    err = (pd.to_numeric(numeric_ans["value"], errors="coerce")
           - pd.to_numeric(numeric_ans["clean_value"], errors="coerce")).abs()
    mask = np.isfinite(tol) & (tol > 0) & np.isfinite(err)
    return float((err[mask] / tol[mask]).mean()) if mask.any() else float("nan")


def proportion_cis(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float, int]]:
    """Wilson 95% intervals for the proportion-valued headline metrics.

    Bootstrap CIs on all-zero/all-one outcomes are degenerate ([0,0] or [1,1]);
    Wilson intervals give honest nonzero-width bounds (e.g. 0/60 -> [0, 0.06]).
    Returns metric -> (lo, hi, n).
    """
    from . import selective
    df = pd.DataFrame(rows)
    ans = df[df["answered"]]
    biased = df[df["is_biased"]]
    biased_conf = biased[biased["answered"] & (biased["confidence"] >= 0.5)]
    interval_ans = ans[
        np.isfinite(pd.to_numeric(ans["ci_lower"], errors="coerce"))
        & np.isfinite(pd.to_numeric(ans["ci_upper"], errors="coerce"))
        & np.isfinite(pd.to_numeric(ans["clean_value"], errors="coerce"))
    ] if len(ans) else ans
    out = {}
    if len(biased):
        k = int((~biased_conf["correct"]).sum()) if len(biased_conf) else 0
        lo, hi = selective.wilson_interval(k, len(biased))
        out["confidently_wrong_on_biased"] = (lo, hi, len(biased))
    if len(interval_ans):
        lo, hi = selective.wilson_interval(int(interval_ans["ci_covers_clean"].sum()), len(interval_ans))
        out["interval_coverage_of_clean"] = (lo, hi, len(interval_ans))
    if len(ans):
        lo, hi = selective.wilson_interval(int(ans["correct"].sum()), len(ans))
        out["accuracy_answered"] = (lo, hi, len(ans))
    return out


def cluster_bootstrap_ci(rows: list[dict[str, Any]], metric: str,
                         cluster_key: str = "domain", n_boot: int = 2000,
                         seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Cluster bootstrap 95% CI: resample whole domain instances, not
    questions, since questions within an instance share records and anchor
    (question-level resampling would pseudoreplicate)."""
    rng = np.random.default_rng(seed)
    clusters: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        clusters.setdefault(r.get(cluster_key), []).append(r)
    keys = list(clusters)
    if not keys:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        sample: list[dict[str, Any]] = []
        for i in pick:
            sample.extend(clusters[keys[i]])
        v = score(sample).get(metric, float("nan"))
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return (float(np.quantile(vals, alpha / 2.0)),
            float(np.quantile(vals, 1.0 - alpha / 2.0)))


def bootstrap_ci(rows: list[dict[str, Any]], metric: str, n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Question-level nonparametric bootstrap 95% CI for one score() metric."""
    rng = np.random.default_rng(seed)
    n = len(rows)
    if n == 0:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = score([rows[i] for i in idx]).get(metric, float("nan"))
        if v == v:  # finite / not-nan
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return (float(np.quantile(vals, alpha / 2.0)),
            float(np.quantile(vals, 1.0 - alpha / 2.0)))


def bootstrap_table(system_rows: dict[str, list[dict[str, Any]]],
                    metrics: list[str], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Point estimate + bootstrap 95% CI for each system x metric."""
    recs = []
    for name, rows in system_rows.items():
        s = score(rows)
        for m in metrics:
            lo, hi = bootstrap_ci(rows, m, n_boot=n_boot, seed=seed)
            recs.append({"system": name, "metric": m, "value": s.get(m),
                         "ci95_lo": lo, "ci95_hi": hi})
    return pd.DataFrame(recs)


def score_by(rows: list[dict[str, Any]], key: str) -> pd.DataFrame:
    """Per-group (e.g. per-domain or per-type) scores for one system."""
    df = pd.DataFrame(rows)
    out = []
    for val, g in df.groupby(key):
        s = score(g.to_dict("records"))
        s[key] = val
        out.append(s)
    cols = [key] + [c for c in out[0] if c != key] if out else [key]
    return pd.DataFrame(out)[cols] if out else pd.DataFrame(columns=[key])


def compare(naive_rows: list[dict[str, Any]], bias_aware_rows: list[dict[str, Any]]) -> pd.DataFrame:
    metrics = [
        ("confidently_wrong_on_biased", "lower"),
        ("interval_coverage_of_clean", "higher"),
        ("ece", "lower"),
        ("aurc_biased_subset", "lower"),
        ("mean_abs_error_answered", "lower"),
        ("accuracy_answered", "higher"),
        ("coverage", "higher"),
    ]
    s_naive, s_ba = score(naive_rows), score(bias_aware_rows)
    rows = []
    for m, better in metrics:
        rows.append({
            "metric": m,
            "better_when": better,
            "naive_compass": round(s_naive[m], 4),
            "bias_aware": round(s_ba[m], 4),
        })
    return pd.DataFrame(rows)
