"""Backtesting and player prop accuracy evaluation (Phase 4).

For each player-game in the target season, the model predicts the Q4
per-minute drop-off from Q1-Q3 pace.  That drop-off is added to the
player's observed Q1-Q3 rate to produce a Q4 stat prediction, which is
then compared against:
  1. The naive baseline — zero drop-off (Q4 rate = Q1-Q3 rate)
  2. The actual observed Q4 outcome

Two evaluation layers
---------------------
  Per-minute MAE — tracks how well the model predicts efficiency changes.
  Prop accuracy  — simulates an over/under bet on Q4 total stats at common
                   thresholds; measures directional accuracy improvement.

Realistic expectation
---------------------
Fatigue is a real but small effect (R² ≈ 0.05-0.20 with full data).
Expect modest MAE improvement (~1-5%) over the naive baseline.  Large
improvements suggest overfitting; zero improvement on all targets with full
data suggests broken features or labels.

Usage
-----
>>> from src.evaluate import run_backtest
>>> summary = run_backtest(season="2024-25")
>>> print(summary.to_string())
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "feature_matrix.parquet"
MODEL_PATH    = PROJECT_ROOT / "data" / "fatigue_model.pkl"
DATA_DIR      = PROJECT_ROOT / "data"
FIGURES_DIR   = PROJECT_ROOT / "figures"

from src.model import FEATURE_COLS, TARGET_COLS, TRAIN_SEASONS, TEST_SEASON

# Q4 prop thresholds (total points scored in Q4)
_PROP_THRESHOLDS_PTS = [3, 5, 8, 10, 12, 15]
_PROP_THRESHOLDS_AST = [1, 2, 3, 4]

# Minimum Q4 minutes to include a player-game in prop analysis
_MIN_Q4_MINUTES = 3.0


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(season: str = "2024-25") -> pd.DataFrame:
    """Run game-by-game Q4 prediction backtest for the specified season.

    For each player-game:
      - fatigue prediction  = Q1-Q3 per-min rate + model drop-off
      - naive prediction    = Q1-Q3 per-min rate (zero drop-off)
      - actual              = observed Q4 per-min rate

    Converts per-minute predictions to total Q4 stats using actual Q4
    minutes (oracle backtest — maximises signal, see Notes).

    Parameters
    ----------
    season:
        NBA season string, e.g. ``"2024-25"``.

    Returns
    -------
    summary : DataFrame
        One row per archetype segment with columns:
        n, pts_fatigue_mae, pts_naive_mae, pts_imprv_pct,
        ast_fatigue_mae, ast_naive_mae, ast_imprv_pct

    Notes
    -----
    Using actual Q4 minutes (oracle) is appropriate for a research
    backtest — it isolates the model's ability to predict drop-off from
    noise introduced by playing-time changes.  For a live prop model you
    would substitute estimated Q4 minutes (e.g. Q1-Q3 avg / 3).
    """
    # ── Load artefacts ─────────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run first:  python predict.py --train"
        )
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Feature matrix not found. Run first:  python predict.py --features"
        )

    with open(MODEL_PATH, "rb") as fh:
        artifact = pickle.load(fh)
    models = artifact["models"]

    feat = pd.read_parquet(FEATURES_PATH)

    # ── Select backtest rows ───────────────────────────────────────────────
    if "SEASON" in feat.columns and season in feat["SEASON"].unique():
        backtest_mask = feat["SEASON"] == season
        avail_train   = set(feat["SEASON"].unique()) - {season}
        if avail_train.issubset(set(TRAIN_SEASONS)):
            logger.info("Using canonical season split: test = %s", season)
        else:
            logger.warning(
                "Season %s found but training seasons may overlap. "
                "Results may be optimistic.",
                season,
            )
    else:
        # Fall back to the last 20% of whatever is cached
        logger.warning(
            "Season '%s' not in feature matrix — backtesting on last 20%% of cache.",
            season,
        )
        n = len(feat)
        backtest_mask = feat.index >= int(n * 0.80)

    sub = feat[backtest_mask].copy().reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No rows found for season '{season}'.")

    logger.info("Backtest: %d player-games", len(sub))

    # ── Build per-minute predictions ───────────────────────────────────────
    results = _build_predictions(sub, models)

    # ── Save full results ──────────────────────────────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"backtest_{season.replace('-', '_')}.parquet"
    results.to_parquet(out_path, index=False)
    logger.info("Full backtest saved -> %s", out_path)

    # ── MAE summary by archetype ───────────────────────────────────────────
    summary = _mae_summary(results)

    # ── Prop accuracy ──────────────────────────────────────────────────────
    pts_prop = prop_accuracy(results, stat="pts", thresholds=_PROP_THRESHOLDS_PTS)
    ast_prop = prop_accuracy(results, stat="ast", thresholds=_PROP_THRESHOLDS_AST)

    _print_backtest_results(summary, pts_prop, ast_prop, season, len(sub))

    return summary


def prop_accuracy(
    results: pd.DataFrame,
    stat: str = "pts",
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate over/under accuracy at common prop thresholds.

    Simulates a bet where you predict whether a player's total Q4 stat
    will exceed a threshold.  Fatigue-adjusted vs naive accuracy are both
    reported.

    Parameters
    ----------
    results:
        Output of ``_build_predictions()``.  Must contain columns
        ``{stat}_pred_fatigue``, ``{stat}_pred_naive``, ``{stat}_actual``.
    stat:
        Stat to evaluate: ``"pts"`` or ``"ast"``.
    thresholds:
        Prop line values to test.

    Returns
    -------
    DataFrame with columns:
        threshold, fatigue_acc, naive_acc, delta_acc, n
    """
    if thresholds is None:
        thresholds = _PROP_THRESHOLDS_PTS

    fat_col    = f"{stat}_pred_fatigue"
    naive_col  = f"{stat}_pred_naive"
    actual_col = f"{stat}_actual"

    required = [fat_col, naive_col, actual_col]
    missing  = [c for c in required if c not in results.columns]
    if missing:
        raise ValueError(f"prop_accuracy: missing columns {missing}")

    valid = results.dropna(subset=required)
    rows  = []

    for t in thresholds:
        fat_over    = valid[fat_col]    > t
        naive_over  = valid[naive_col]  > t
        actual_over = valid[actual_col] > t

        fat_acc   = (fat_over   == actual_over).mean()
        naive_acc = (naive_over == actual_over).mean()

        rows.append({
            "threshold":   t,
            "fatigue_acc": round(float(fat_acc),   4),
            "naive_acc":   round(float(naive_acc), 4),
            "delta_acc":   round(float(fat_acc - naive_acc), 4),
            "n":           len(valid),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _build_predictions(sub: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Apply each model to X and construct the full results DataFrame.

    Columns produced
    ----------------
    For each stat in {pts, ast, to, fg_pct}:
      {stat}_rate_q1q3          Q1-Q3 per-minute rate (the naive Q4 prediction)
      {stat}_dropoff_pred        model's predicted Q4 drop-off
      {stat}_rate_pred_fatigue   Q1Q3 rate + predicted drop-off
      {stat}_rate_actual         observed Q4 per-minute rate
      {stat}_pred_fatigue        total Q4 stat (rate × actual Q4 minutes)
      {stat}_pred_naive          total Q4 stat using naive rate
      {stat}_actual              total Q4 stat actually observed
    """
    X = sub[FEATURE_COLS].copy()

    out = sub[["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "GAME_DATE", "SEASON",
               "minutes_q4", "is_home", "rest_days", "is_back_to_back",
               "START_POSITION", "player_age"]].copy()

    stat_map = {
        "q4_pts_dropoff":    ("pts_per_min_q1q3",  "pts_per_min_q4",  "pts_q4"),
        "q4_ast_dropoff":    ("ast_per_min_q1q3",  "ast_per_min_q4",  "ast_q4"),
        "q4_to_surplus":     ("to_per_min_q1q3",   "to_per_min_q4",   "to_q4"),
        "q4_fg_pct_dropoff": ("fg_pct_q1q3",       "fg_pct_q4",       None),
    }

    stat_short = {
        "q4_pts_dropoff":    "pts",
        "q4_ast_dropoff":    "ast",
        "q4_to_surplus":     "to",
        "q4_fg_pct_dropoff": "fg_pct",
    }

    for target, model in models.items():
        q1q3_rate_col, q4_rate_col, q4_total_col = stat_map[target]
        s = stat_short[target]

        drop_pred = model.predict(X)

        q1q3_rate = sub[q1q3_rate_col].values if q1q3_rate_col in sub.columns else np.zeros(len(sub))
        q4_actual_rate = sub[q4_rate_col].values if q4_rate_col in sub.columns else np.full(len(sub), np.nan)

        out[f"{s}_rate_q1q3"]         = q1q3_rate
        out[f"{s}_dropoff_pred"]       = drop_pred
        out[f"{s}_rate_pred_fatigue"]  = q1q3_rate + drop_pred
        out[f"{s}_rate_actual"]        = q4_actual_rate

        # Convert per-minute → totals using actual Q4 minutes
        q4_min = sub["minutes_q4"].values.clip(min=0)
        out[f"{s}_pred_fatigue"] = (q1q3_rate + drop_pred) * q4_min
        out[f"{s}_pred_naive"]   = q1q3_rate * q4_min

        if q4_total_col and q4_total_col in sub.columns:
            out[f"{s}_actual"] = sub[q4_total_col].values
        else:
            out[f"{s}_actual"] = q4_actual_rate * q4_min

    # Segment tags
    if "START_POSITION" in out.columns:
        out["is_starter"] = out["START_POSITION"].notna() & (out["START_POSITION"] != "")
    if "player_age" in out.columns:
        out["age_group"] = pd.cut(
            out["player_age"],
            bins=[0, 25, 32, 99],
            labels=["Young (<25)", "Prime (25-31)", "Veteran (>=32)"],
        )

    return out


def _mae_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Compute per-minute MAE for fatigue vs naive, broken down by archetype."""
    rows = []

    def _row(label: str, mask: pd.Series) -> dict:
        sub = results[mask]
        n   = int(mask.sum())
        if n < 5:
            return {}

        entry: dict = {"archetype": label, "n": n}
        for stat, rate_col in [("pts", "pts_rate"), ("ast", "ast_rate")]:
            fat_col    = f"{stat}_rate_pred_fatigue"
            naive_col  = f"{stat}_rate_q1q3"
            actual_col = f"{stat}_rate_actual"

            if not all(c in sub.columns for c in [fat_col, naive_col, actual_col]):
                continue

            valid = sub[[fat_col, naive_col, actual_col]].dropna()
            if len(valid) < 5:
                continue

            fat_mae   = mean_absolute_error(valid[actual_col], valid[fat_col])
            naive_mae = mean_absolute_error(valid[actual_col], valid[naive_col])
            imprv     = (naive_mae - fat_mae) / naive_mae * 100 if naive_mae > 0 else 0.0

            entry[f"{stat}_fatigue_mae"] = round(fat_mae,   4)
            entry[f"{stat}_naive_mae"]   = round(naive_mae, 4)
            entry[f"{stat}_imprv_pct"]   = round(imprv,     2)

        return entry

    # Overall
    rows.append(_row("All players", pd.Series(True, index=results.index)))

    # Role
    if "is_starter" in results.columns:
        rows.append(_row("Starters", results["is_starter"] == True))
        rows.append(_row("Bench",    results["is_starter"] == False))

    # Age group
    if "age_group" in results.columns:
        for grp in ["Young (<25)", "Prime (25-31)", "Veteran (>=32)"]:
            rows.append(_row(grp, results["age_group"] == grp))

    # Fatigue context
    if "is_back_to_back" in results.columns:
        rows.append(_row("Back-to-back", results["is_back_to_back"] == True))
        rows.append(_row("Rested",       results["is_back_to_back"] == False))

    valid_rows = [r for r in rows if r]
    if not valid_rows:
        return pd.DataFrame()

    return pd.DataFrame(valid_rows).set_index("archetype")


# ═══════════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════════

def _print_backtest_results(
    summary: pd.DataFrame,
    pts_prop: pd.DataFrame,
    ast_prop: pd.DataFrame,
    season: str,
    n_games: int,
) -> None:
    sep = "=" * 66

    # ── MAE summary ────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"BACKTEST RESULTS — {season}  ({n_games:,} player-games)")
    print(sep)
    print(f"\n  Per-minute MAE  (fatigue model vs naive zero-drop-off)\n")

    has_pts = "pts_fatigue_mae" in summary.columns
    has_ast = "ast_fatigue_mae" in summary.columns

    if has_pts and has_ast:
        hdr = f"  {'Archetype':<22}  {'N':>5}  {'PTS fat':>8}  {'PTS nv':>7}  {'Imprv%':>7}  {'AST fat':>8}  {'AST nv':>7}  {'Imprv%':>7}"
        print(hdr)
        print(f"  {'-'*22}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}")
        for arch, row in summary.iterrows():
            pts_i = row.get("pts_imprv_pct", 0)
            ast_i = row.get("ast_imprv_pct", 0)
            pts_flag = "+" if pts_i > 0 else ""
            ast_flag = "+" if ast_i > 0 else ""
            print(
                f"  {str(arch):<22}  {int(row['n']):>5}  "
                f"{row.get('pts_fatigue_mae', 0):>8.4f}  {row.get('pts_naive_mae', 0):>7.4f}  "
                f"{pts_flag}{pts_i:>6.1f}%  "
                f"{row.get('ast_fatigue_mae', 0):>8.4f}  {row.get('ast_naive_mae', 0):>7.4f}  "
                f"{ast_flag}{ast_i:>6.1f}%"
            )

    # ── Prop accuracy — points ─────────────────────────────────────────────
    if not pts_prop.empty:
        print(f"\n  Q4 Points prop accuracy  (over/under)\n")
        print(f"  {'Threshold':>10}  {'Fatigue':>8}  {'Naive':>7}  {'Delta':>7}  {'N':>5}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*5}")
        for _, r in pts_prop.iterrows():
            delta = r["delta_acc"]
            sign  = "+" if delta >= 0 else ""
            print(
                f"  {'>' + str(int(r['threshold'])) + ' pts':>10}  "
                f"{r['fatigue_acc']:.1%}  {r['naive_acc']:.1%}  "
                f"{sign}{delta:.1%}  {int(r['n']):>5}"
            )

    # ── Prop accuracy — assists ────────────────────────────────────────────
    if not ast_prop.empty:
        print(f"\n  Q4 Assists prop accuracy  (over/under)\n")
        print(f"  {'Threshold':>10}  {'Fatigue':>8}  {'Naive':>7}  {'Delta':>7}  {'N':>5}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*5}")
        for _, r in ast_prop.iterrows():
            delta = r["delta_acc"]
            sign  = "+" if delta >= 0 else ""
            print(
                f"  {'>' + str(int(r['threshold'])) + ' ast':>10}  "
                f"{r['fatigue_acc']:.1%}  {r['naive_acc']:.1%}  "
                f"{sign}{delta:.1%}  {int(r['n']):>5}"
            )

    print(f"\n{sep}\n")
