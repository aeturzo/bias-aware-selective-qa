from __future__ import annotations

from pathlib import Path

import pandas as pd


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_uq_intervals(results: pd.DataFrame, outpath: Path) -> None:
    if results.empty:
        return
    plt = _setup_matplotlib()
    df = results[results["quantity"] == "carbon_gap"].copy()
    if df.empty:
        df = results.copy()
    df = df.sort_values(["method", "seed"]).groupby("method", as_index=False).agg(
        posterior_mean=("posterior_mean", "mean"),
        ci_lower=("ci_lower", "mean"),
        ci_upper=("ci_upper", "mean"),
        clean_value=("clean_value", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = range(len(df))
    ax.errorbar(
        df["posterior_mean"],
        list(y),
        xerr=[
            df["posterior_mean"] - df["ci_lower"],
            df["ci_upper"] - df["posterior_mean"],
        ],
        fmt="o",
        capsize=3,
    )
    if not df.empty:
        ax.axvline(float(df["clean_value"].iloc[0]), color="black", linestyle="--", linewidth=1, label="clean")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["method"])
    ax.set_xlabel("Carbon gap credible interval")
    ax.set_title("Synthetic UQ intervals")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_lambda_trajectories(lambda_df: pd.DataFrame, outpath: Path) -> None:
    if lambda_df.empty:
        return
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    grouped = lambda_df.groupby(["method", "constraint", "round"], as_index=False)["lambda_after"].mean()
    for (method, constraint), gdf in grouped.groupby(["method", "constraint"]):
        ax.plot(gdf["round"], gdf["lambda_after"], marker="o", linewidth=1.5, label=f"{method}:{constraint}")
    ax.set_xlabel("Adaptation round")
    ax.set_ylabel("Lambda")
    ax.set_title("Constraint calibration")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_coverage_vs_width(summary: pd.DataFrame, outpath: Path) -> None:
    if summary.empty:
        return
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for quantity, gdf in summary.groupby("quantity"):
        ax.scatter(gdf["ci_width"], gdf["ci_covers_clean"], label=quantity)
        for _, row in gdf.iterrows():
            ax.annotate(str(row["method"]), (row["ci_width"], row["ci_covers_clean"]), fontsize=7, alpha=0.75)
    ax.set_xlabel("Mean CI width")
    ax.set_ylabel("Clean-value coverage rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Coverage vs interval width")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_fairness_vs_width(summary: pd.DataFrame, outpath: Path) -> None:
    if summary.empty:
        return
    plt = _setup_matplotlib()
    df = summary[summary["quantity"] == "carbon_gap"].copy()
    if df.empty:
        df = summary.copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.scatter(df["ci_width"], df["d_clean_mean"])
    for _, row in df.iterrows():
        ax.annotate(str(row["method"]), (row["ci_width"], row["d_clean_mean"]), fontsize=8)
    ax.set_xlabel("Mean CI width")
    ax.set_ylabel("Mean distance to clean gaps")
    ax.set_title("Fairness-error vs interval width")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_ess_comparison(summary: pd.DataFrame, outpath: Path) -> None:
    if summary.empty:
        return
    plt = _setup_matplotlib()
    df = summary[summary["quantity"] == "carbon_gap"].copy()
    if df.empty:
        df = summary.copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(df["method"], df["ESS"])
    ax.set_ylabel("ESS")
    ax.set_title("Effective sample size comparison")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_synthetic_plots(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    lambda_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_uq_intervals(results, figures_dir / "uq_intervals_synthetic.png")
    plot_lambda_trajectories(lambda_df, figures_dir / "lambda_trajectories.png")
    plot_coverage_vs_width(summary, figures_dir / "coverage_vs_width.png")
    plot_fairness_vs_width(summary, figures_dir / "fairness_vs_width.png")
    plot_ess_comparison(summary, figures_dir / "ess_comparison.png")

