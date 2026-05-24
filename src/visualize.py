"""Publication-quality fatigue visualization (Phase 5).

Generates five diagnostic plots saved to figures/:
  fatigue_curve.png      — FG% by quarter for high- vs low-minute players
  dropoff_heatmap.png    — Q4 drop-off by (cumulative minutes × rest days)
  feature_importance.png — XGBoost feature importance bar chart
  pred_vs_actual.png     — Holdout prediction vs actual scatter
  back_to_back.png       — Performance comparison on B2B vs rested games

All plots use a consistent style (dark background, CMU research-report ready).

Usage
-----
>>> from src.visualize import generate_all
>>> generate_all()
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = DATA_DIR / "fatigue_model.pkl"
FEATURE_MATRIX_PATH = DATA_DIR / "feature_matrix.parquet"
BACKTEST_PATH = DATA_DIR / "backtest_2024_25.parquet"
BOX_SCORES_DIR = DATA_DIR / "box_scores"

# Shared style
PALETTE = {"high_load": "#E85D04", "low_load": "#0077B6", "neutral": "#6C757D"}
STYLE = "dark_background"
DPI = 150


def _setup_style() -> None:
    plt.style.use(STYLE)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    logger.info(f"Saved -> {out}")


def generate_all(force: bool = False) -> None:
    """Generate all five diagnostic plots and save to figures/."""
    _setup_style()

    fm = pd.read_parquet(FEATURE_MATRIX_PATH)
    bt = pd.read_parquet(BACKTEST_PATH) if BACKTEST_PATH.exists() else None

    artifact = None
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as fh:
            artifact = pickle.load(fh)

    fig1 = plot_fatigue_curve(save=True)
    plt.close(fig1)

    fig2 = plot_dropoff_heatmap(fm, save=True)
    plt.close(fig2)

    if artifact is not None:
        fig3 = plot_feature_importance(artifact, save=True)
        plt.close(fig3)
    else:
        logger.warning("No model artifact found — skipping feature_importance.png")

    if bt is not None:
        fig4 = plot_pred_vs_actual(bt, save=True)
        plt.close(fig4)
    else:
        logger.warning("No backtest file found — skipping pred_vs_actual.png")

    fig5 = plot_back_to_back(fm, save=True)
    plt.close(fig5)

    print(f"All plots saved to {FIGURES_DIR}/")


# ---------------------------------------------------------------------------
# Plot 1 — Fatigue curve (FG% by quarter, high vs low load)
# ---------------------------------------------------------------------------

def plot_fatigue_curve(save: bool = True) -> plt.Figure:
    """FG% by quarter split by cumulative minutes load (high vs low).

    High load = top tercile of Q1-Q3 minutes; low load = bottom tercile.
    Uses raw box_scores/ parquet files for true per-quarter breakdown.
    """
    _setup_style()

    # Load and concatenate all box score files
    box_files = sorted(BOX_SCORES_DIR.glob("*.parquet"))
    if not box_files:
        raise FileNotFoundError(f"No box score files found in {BOX_SCORES_DIR}")

    parts = []
    for f in box_files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception:
            pass
    raw = pd.concat(parts, ignore_index=True)

    # Normalize column names
    raw.columns = [c.upper() for c in raw.columns]

    # Keep only regulation quarters
    raw = raw[raw["PERIOD"].isin([1, 2, 3, 4])].copy()
    raw["FGM"] = pd.to_numeric(raw["FGM"], errors="coerce").fillna(0)
    raw["FGA"] = pd.to_numeric(raw["FGA"], errors="coerce").fillna(0)
    raw["MIN_DECIMAL"] = pd.to_numeric(raw.get("MIN_DECIMAL", pd.Series(dtype=float)), errors="coerce").fillna(0)

    # Compute cumulative Q1-Q3 minutes per player-game
    q1q3 = raw[raw["PERIOD"].isin([1, 2, 3])].groupby(
        ["GAME_ID", "PLAYER_ID"], as_index=False
    ).agg(cum_min=("MIN_DECIMAL", "sum"), fgm_q1q3=("FGM", "sum"), fga_q1q3=("FGA", "sum"))

    # Tercile thresholds on Q1-Q3 minutes
    lo_thresh = q1q3["cum_min"].quantile(1 / 3)
    hi_thresh = q1q3["cum_min"].quantile(2 / 3)

    high_pairs = set(
        zip(
            q1q3.loc[q1q3["cum_min"] >= hi_thresh, "GAME_ID"],
            q1q3.loc[q1q3["cum_min"] >= hi_thresh, "PLAYER_ID"],
        )
    )
    low_pairs = set(
        zip(
            q1q3.loc[q1q3["cum_min"] <= lo_thresh, "GAME_ID"],
            q1q3.loc[q1q3["cum_min"] <= lo_thresh, "PLAYER_ID"],
        )
    )

    raw["pair"] = list(zip(raw["GAME_ID"], raw["PLAYER_ID"]))
    raw["load_group"] = np.where(
        raw["pair"].isin(high_pairs),
        "High load (top tercile)",
        np.where(raw["pair"].isin(low_pairs), "Low load (bottom tercile)", "mid"),
    )
    plot_df = raw[raw["load_group"] != "mid"].copy()

    # Aggregate FG% per quarter per group
    agg = (
        plot_df.groupby(["load_group", "PERIOD"])
        .agg(fgm=("FGM", "sum"), fga=("FGA", "sum"))
        .reset_index()
    )
    agg["fg_pct"] = np.where(agg["fga"] > 0, agg["fgm"] / agg["fga"] * 100, np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"High load (top tercile)": PALETTE["high_load"], "Low load (bottom tercile)": PALETTE["low_load"]}

    for group, gdf in agg.groupby("load_group"):
        gdf = gdf.sort_values("PERIOD")
        ax.plot(
            gdf["PERIOD"],
            gdf["fg_pct"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            color=colors[group],
            label=group,
        )

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel("FG% (%)", fontsize=12)
    ax.set_title("FG% by Quarter: High-Load vs Low-Load Players", fontsize=14, pad=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save:
        _save(fig, "fatigue_curve.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — Drop-off heatmap (cumulative minutes x rest days)
# ---------------------------------------------------------------------------

def plot_dropoff_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Heatmap of Q4 scoring drop-off by (cumulative minutes x rest days)."""
    _setup_style()

    work = df[["cumulative_minutes_q1q3", "rest_days", "q4_pts_dropoff"]].dropna().copy()

    # Bin cumulative minutes into 4 buckets
    work["min_bin"] = pd.cut(
        work["cumulative_minutes_q1q3"],
        bins=4,
        labels=["<18 min", "18-24 min", "24-29 min", "29+ min"],
    )

    # Cap rest_days at 7 for readability
    work["rest_cap"] = work["rest_days"].clip(upper=7).astype(int)

    pivot = work.groupby(["min_bin", "rest_cap"], observed=False)["q4_pts_dropoff"].mean().unstack("rest_cap")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Avg Q4 pts/min drop-off"},
    )
    ax.set_xlabel("Rest Days (capped at 7)", fontsize=12)
    ax.set_ylabel("Cumulative Minutes Q1-Q3", fontsize=12)
    ax.set_title("Q4 Scoring Drop-Off by Load & Rest", fontsize=14, pad=12)
    fig.tight_layout()

    if save:
        _save(fig, "dropoff_heatmap.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(artifact: dict, save: bool = True) -> plt.Figure:
    """Horizontal bar chart of XGBoost feature importances (gain)."""
    _setup_style()

    fi = artifact.get("feature_importance")
    if fi is None or fi.empty:
        raise ValueError("No feature_importance in model artifact.")

    # fi has features as index; pick mean_gain or fallback to first numeric col
    if "mean_gain" in fi.columns:
        gain_series = fi["mean_gain"]
    elif "gain" in fi.columns:
        gain_series = fi["gain"]
    else:
        gain_series = fi.select_dtypes("number").mean(axis=1)

    agg = gain_series.rename("gain").reset_index()
    agg.columns = ["feature", "gain"]
    agg = agg.sort_values("gain", ascending=True).tail(15)  # top 15

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(agg["feature"], agg["gain"], color=PALETTE["low_load"], alpha=0.85)

    # Highlight top bar
    bars[-1].set_color(PALETTE["high_load"])

    ax.set_xlabel("Mean Gain (across targets)", fontsize=12)
    ax.set_title("Top XGBoost Feature Importances", fontsize=14, pad=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if save:
        _save(fig, "feature_importance.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 4 — Pred vs Actual scatter
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(bt: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Scatter plot of predicted vs actual Q4 pts/min drop-off on holdout."""
    _setup_style()

    needed = {"pts_dropoff_pred", "pts_rate_actual", "pts_rate_q1q3"}
    missing = needed - set(bt.columns)
    if missing:
        raise KeyError(f"Backtest DataFrame missing columns: {missing}")

    sub = bt[list(needed)].dropna().copy()
    sub["actual_dropoff"] = sub["pts_rate_actual"] - sub["pts_rate_q1q3"]
    pred = sub["pts_dropoff_pred"]
    actual = sub["actual_dropoff"]

    corr = np.corrcoef(pred, actual)[0, 1] if len(pred) > 1 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 7))

    # Scatter colored by actual dropoff magnitude
    sc = ax.scatter(
        pred,
        actual,
        c=actual,
        cmap="RdYlGn_r",
        alpha=0.65,
        edgecolors="none",
        s=40,
    )
    plt.colorbar(sc, ax=ax, label="Actual drop-off (pts/min)")

    # Identity line
    lo = min(pred.min(), actual.min())
    hi = max(pred.max(), actual.max())
    ax.plot([lo, hi], [lo, hi], "--", color=PALETTE["neutral"], linewidth=1.5, label="Perfect")

    # Zero-line references
    ax.axhline(0, color="white", linewidth=0.6, alpha=0.3)
    ax.axvline(0, color="white", linewidth=0.6, alpha=0.3)

    ax.set_xlabel("Predicted Q4 pts/min drop-off", fontsize=12)
    ax.set_ylabel("Actual Q4 pts/min drop-off", fontsize=12)
    ax.set_title(
        f"Predicted vs Actual Q4 Drop-off  (r = {corr:.3f})", fontsize=14, pad=12
    )
    ax.legend(fontsize=10)
    fig.tight_layout()

    if save:
        _save(fig, "pred_vs_actual.png")
    return fig


# ---------------------------------------------------------------------------
# Plot 5 — Back-to-back comparison
# ---------------------------------------------------------------------------

def plot_back_to_back(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Box plots comparing Q4 pts drop-off: B2B vs 1 rest day vs 2+ rest days."""
    _setup_style()

    work = df[["rest_days", "q4_pts_dropoff"]].dropna().copy()

    def _rest_label(d: int) -> str:
        if d == 0:
            return "B2B (0 rest)"
        if d == 1:
            return "1 rest day"
        return "2+ rest days"

    work["rest_cat"] = work["rest_days"].apply(_rest_label)

    order = ["B2B (0 rest)", "1 rest day", "2+ rest days"]
    colors = [PALETTE["high_load"], PALETTE["neutral"], PALETTE["low_load"]]

    fig, ax = plt.subplots(figsize=(8, 6))

    groups = [work.loc[work["rest_cat"] == cat, "q4_pts_dropoff"].dropna().values for cat in order]

    bp = ax.boxplot(
        groups,
        labels=order,
        patch_artist=True,
        notch=False,
        widths=0.5,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": "white"},
        capprops={"color": "white"},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Overlay strip
    for i, (cat, color) in enumerate(zip(order, colors), start=1):
        vals = work.loc[work["rest_cat"] == cat, "q4_pts_dropoff"].dropna().values
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=color,
            alpha=0.35,
            s=18,
            zorder=3,
        )

    # Annotate sample sizes
    for i, (cat, vals) in enumerate(zip(order, groups), start=1):
        ax.text(i, ax.get_ylim()[0], f"n={len(vals)}", ha="center", va="bottom", fontsize=9, color="white")

    ax.axhline(0, color=PALETTE["neutral"], linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("Rest Category", fontsize=12)
    ax.set_ylabel("Q4 pts/min drop-off", fontsize=12)
    ax.set_title("Q4 Scoring Drop-Off: B2B vs Rested Games", fontsize=14, pad=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if save:
        _save(fig, "back_to_back.png")
    return fig
