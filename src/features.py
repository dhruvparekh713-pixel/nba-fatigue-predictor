"""Fatigue feature engineering (Phase 2).

Transforms the raw player-game-quarter DataFrame from Phase 1 into a
player-game-level feature matrix ready for model training.

One output row per qualifying player-game:
  • must have ≥ MIN_Q4_MINUTES   minutes in Q4  (excludes garbage time)
  • must have ≥ MIN_Q1Q3_MINUTES minutes in Q1-Q3 (excludes sporadic appearances)

Column conventions
------------------
*_q1q3        aggregated/averaged over Q1-Q3 (pre-fatigue context)
*_q4          Q4 value (fatigue outcome / target)
*_trend       OLS slope of a per-quarter series over Q1→Q2→Q3
              positive = increasing, negative = decreasing

Target variables (y)
--------------------
q4_pts_dropoff     (Q4 pts/min)  − (Q1-Q3 pts/min)
q4_fg_pct_dropoff  Q4 FG%        − Q1-Q3 FG%
q4_ast_dropoff     (Q4 ast/min)  − (Q1-Q3 ast/min)
q4_to_surplus      (Q4 TO/min)   − (Q1-Q3 TO/min)   [positive = worse]

Usage
-----
>>> from src.features import build_feature_matrix
>>> X, y = build_feature_matrix()            # uses cached data/player_game_quarters_all.parquet
>>> X, y = build_feature_matrix(force_refresh=True)  # recompute
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
FEATURES_PATH  = DATA_DIR / "feature_matrix.parquet"
PLAYER_INFO_PATH = DATA_DIR / "player_info.parquet"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_Q4_MINUTES: float   = 3.0   # minimum Q4 minutes to keep a player-game
MIN_Q1Q3_MINUTES: float = 10.0  # minimum Q1-Q3 minutes to keep a player-game

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "cumulative_minutes_q1q3",   # total minutes through end of Q3
    "pace_q1q3",                 # player's team: possessions/48 in Q1-Q3
    "usage_trend",               # OLS slope of usage proxy across Q1→Q3
    "fg_pct_trend",              # OLS slope of FG% across Q1→Q3
    "rest_days",                 # days since last game  (0 = back-to-back)
    "season_minutes_load",       # cumulative season minutes entering this game
    "game_pace",                 # both teams' average pace in Q1-Q3
    "score_diff_entering_q4",    # team Q1-Q3 pts minus opponent Q1-Q3 pts
    "is_home",                   # 1 = home, 0 = away
    "player_age",                # decimal age at game date
    "minutes_per_game_season_avg",  # rolling avg MPG entering this game
]

TARGET_COLS: list[str] = [
    "q4_pts_dropoff",
    "q4_fg_pct_dropoff",
    "q4_ast_dropoff",
    "q4_to_surplus",
]

ID_COLS: list[str] = [
    "GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
    "SEASON", "GAME_DATE",
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_feature_matrix(
    raw_df: Optional[pd.DataFrame] = None,
    min_q4_minutes: float = MIN_Q4_MINUTES,
    min_q1q3_minutes: float = MIN_Q1Q3_MINUTES,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature matrix (X) and targets (y) for model training.

    Parameters
    ----------
    raw_df:
        Output of ``scraper.build_dataset()``.  If None, loads from
        ``data/player_game_quarters_all.parquet``.
    min_q4_minutes:
        Minimum Q4 playing time to include a player-game.
    min_q1q3_minutes:
        Minimum Q1-Q3 playing time to include a player-game.
    force_refresh:
        Recompute even if ``data/feature_matrix.parquet`` already exists.

    Returns
    -------
    X : DataFrame  shape (N, 11)  — columns = FEATURE_COLS
    y : DataFrame  shape (N, 4)   — columns = TARGET_COLS
    """
    if FEATURES_PATH.exists() and not force_refresh:
        logger.info(f"Loading cached feature matrix ({FEATURES_PATH.name})")
        feat = pd.read_parquet(FEATURES_PATH)
        _log_summary(feat)
        return feat[FEATURE_COLS], feat[TARGET_COLS]

    # ── Load raw data ──────────────────────────────────────────────────────
    if raw_df is None:
        raw_path = DATA_DIR / "player_game_quarters_all.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Raw data not found at {raw_path}.\n"
                "Run Phase 1 first:  python predict.py --scrape"
            )
        logger.info(f"Loading {raw_path.name}…")
        raw_df = pd.read_parquet(raw_path)

    logger.info(
        f"Raw data: {len(raw_df):,} rows | "
        f"{raw_df['GAME_ID'].nunique():,} games | "
        f"{raw_df['PLAYER_ID'].nunique():,} players"
    )

    q1q3 = raw_df[raw_df["PERIOD"].isin([1, 2, 3])].copy()
    q4   = raw_df[raw_df["PERIOD"] == 4].copy()

    # ── Step 1: Player Q1-Q3 aggregates ───────────────────────────────────
    logger.info("Step 1/8 — Q1-Q3 player aggregates…")
    feat = _aggregate_player(q1q3, prefix="q1q3", include_context=True)

    # ── Step 2: Q1→Q3 trend slopes ────────────────────────────────────────
    logger.info("Step 2/8 — Q1→Q3 trend features…")
    trends = _compute_trends(q1q3)
    feat = feat.merge(trends, on=["GAME_ID", "PLAYER_ID"], how="left")

    # ── Step 3: Q4 player aggregates ──────────────────────────────────────
    logger.info("Step 3/8 — Q4 player aggregates…")
    q4_stats = _aggregate_player(q4, prefix="q4", include_context=False)
    _id_cols = {"GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID"}
    q4_extra = [c for c in q4_stats.columns if c not in _id_cols]
    feat = feat.merge(
        q4_stats[["GAME_ID", "PLAYER_ID"] + q4_extra],
        on=["GAME_ID", "PLAYER_ID"],
        how="inner",
    )

    # ── Step 4: Targets ────────────────────────────────────────────────────
    feat = _compute_targets(feat)

    # ── Step 5: Playing-time filter ────────────────────────────────────────
    n_before = len(feat)
    feat = feat[
        (feat["minutes_q1q3"] >= min_q1q3_minutes)
        & (feat["minutes_q4"] >= min_q4_minutes)
    ].copy()
    logger.info(
        f"Step 4/8 — Playing-time filter: {n_before:,} → {len(feat):,} player-games "
        f"({len(feat)/n_before*100:.1f}% retained)"
    )

    # ── Step 6: Team-level features (pace + score diff) ───────────────────
    logger.info("Step 5/8 — Team pace and score differential…")
    team_feats = _compute_team_features(q1q3)
    feat = feat.merge(team_feats, on=["GAME_ID", "TEAM_ID"], how="left")

    # ── Step 7: Season load features ─────────────────────────────────────
    logger.info("Step 6/8 — Season load features…")
    load_feats = _compute_season_load(raw_df)
    feat = feat.merge(load_feats, on=["GAME_ID", "PLAYER_ID"], how="left")

    # ── Step 8: Player ages ───────────────────────────────────────────────
    logger.info("Step 7/8 — Player ages…")
    player_ids = raw_df["PLAYER_ID"].unique().tolist()
    ages = _get_player_ages(player_ids)
    if ages is not None and "birthdate" in ages.columns:
        feat = feat.merge(ages[["PLAYER_ID", "birthdate"]], on="PLAYER_ID", how="left")
        feat["player_age"] = (
            (feat["GAME_DATE"] - feat["birthdate"]).dt.days / 365.25
        ).round(1)
    else:
        feat["player_age"] = np.nan

    # ── Rename / cast context columns ─────────────────────────────────────
    rename_map = {
        "IS_HOME": "is_home",
        "REST_DAYS": "rest_days",
        "IS_BACK_TO_BACK": "is_back_to_back",
        "minutes_q1q3": "cumulative_minutes_q1q3",
    }
    feat = feat.rename(columns={k: v for k, v in rename_map.items() if k in feat.columns})

    if "is_home" in feat.columns:
        feat["is_home"] = feat["is_home"].astype(float)
    if "is_back_to_back" in feat.columns:
        feat["is_back_to_back"] = feat["is_back_to_back"].astype(float)

    # ── Final cleanup (imputation, drop incomplete rows) ──────────────────
    logger.info("Step 8/8 — Imputing missing values…")
    feat = _final_cleanup(feat)

    feat.to_parquet(FEATURES_PATH, index=False)
    _log_summary(feat)

    return feat[FEATURE_COLS], feat[TARGET_COLS]


# ---------------------------------------------------------------------------
# Step 1: Player aggregate stats
# ---------------------------------------------------------------------------

def _aggregate_player(
    df: pd.DataFrame,
    prefix: str,
    include_context: bool,
) -> pd.DataFrame:
    """Aggregate box-score stats for each player-game in df.

    Parameters
    ----------
    prefix:
        Column suffix: ``"q1q3"`` or ``"q4"``.
    include_context:
        If True, also extract game-level context columns (date, rest days, etc.).
    """
    group_cols = ["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]

    agg_spec: dict = {
        f"minutes_{prefix}":  ("MIN_DECIMAL", "sum"),
        f"pts_{prefix}":      ("PTS",     "sum"),
        f"fga_{prefix}":      ("FGA",     "sum"),
        f"fgm_{prefix}":      ("FGM",     "sum"),
        f"fta_{prefix}":      ("FTA",     "sum"),
        f"ftm_{prefix}":      ("FTM",     "sum"),
        f"to_{prefix}":       ("TO",      "sum"),
        f"ast_{prefix}":      ("AST",     "sum"),
        f"reb_{prefix}":      ("REB",     "sum"),
        f"oreb_{prefix}":     ("OREB",    "sum"),
        f"fg3m_{prefix}":     ("FG3M",    "sum"),
        f"fg3a_{prefix}":     ("FG3A",    "sum"),
    }

    if include_context:
        for col in ["GAME_DATE", "SEASON", "IS_HOME", "REST_DAYS",
                    "IS_BACK_TO_BACK", "START_POSITION"]:
            if col in df.columns:
                agg_spec[col] = (col, "first")

    result = df.groupby(group_cols).agg(**agg_spec).reset_index()

    # Per-minute rates (safe: clip denominator to avoid divide-by-zero)
    min_safe = result[f"minutes_{prefix}"].clip(lower=0.5)
    result[f"pts_per_min_{prefix}"]  = result[f"pts_{prefix}"]  / min_safe
    result[f"ast_per_min_{prefix}"]  = result[f"ast_{prefix}"]  / min_safe
    result[f"to_per_min_{prefix}"]   = result[f"to_{prefix}"]   / min_safe
    result[f"fg_pct_{prefix}"] = np.where(
        result[f"fga_{prefix}"] > 0,
        result[f"fgm_{prefix}"] / result[f"fga_{prefix}"],
        np.nan,
    )

    return result


# ---------------------------------------------------------------------------
# Step 2: Q1→Q3 trend features
# ---------------------------------------------------------------------------

def _compute_trends(q1q3: pd.DataFrame) -> pd.DataFrame:
    """Compute OLS slope of usage proxy and FG% across Q1→Q2→Q3.

    Usage proxy per minute = (FGA + 0.44·FTA + TO) / MIN  (proportional to true USG%)
    FG% per quarter        = FGM / FGA  (NaN when FGA = 0)

    The slope captures whether a player is increasing or decreasing their
    scoring/efficiency load as the game progresses — a key fatigue signal.
    """
    per_q = (
        q1q3.groupby(["GAME_ID", "PLAYER_ID", "PERIOD"])
        .agg(
            min_dec=("MIN_DECIMAL", "sum"),
            fga=("FGA",  "sum"),
            fgm=("FGM",  "sum"),
            fta=("FTA",  "sum"),
            to=("TO",    "sum"),
        )
        .reset_index()
    )

    per_q["usage_proxy"] = np.where(
        per_q["min_dec"] > 0.1,
        (per_q["fga"] + 0.44 * per_q["fta"] + per_q["to"])
        / per_q["min_dec"].clip(lower=0.1),
        np.nan,
    )
    per_q["fg_pct"] = np.where(
        per_q["fga"] >= 1,
        per_q["fgm"] / per_q["fga"],
        np.nan,
    )

    # Pivot: (GAME_ID, PLAYER_ID) × PERIOD → wide format
    wide = per_q.pivot_table(
        index=["GAME_ID", "PLAYER_ID"],
        columns="PERIOD",
        values=["usage_proxy", "fg_pct"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_q{int(p)}" for metric, p in wide.columns]
    wide = wide.reset_index()

    # Ensure all expected columns exist (player may not have appeared in a quarter)
    for col in ["usage_proxy_q1", "usage_proxy_q2", "usage_proxy_q3",
                "fg_pct_q1",      "fg_pct_q2",      "fg_pct_q3"]:
        if col not in wide.columns:
            wide[col] = np.nan

    wide["usage_trend"] = _slope_3pt(
        wide["usage_proxy_q1"].values,
        wide["usage_proxy_q2"].values,
        wide["usage_proxy_q3"].values,
    )
    wide["fg_pct_trend"] = _slope_3pt(
        wide["fg_pct_q1"].values,
        wide["fg_pct_q2"].values,
        wide["fg_pct_q3"].values,
    )

    return wide[["GAME_ID", "PLAYER_ID", "usage_trend", "fg_pct_trend"]]


def _slope_3pt(
    y1: np.ndarray, y2: np.ndarray, y3: np.ndarray
) -> np.ndarray:
    """Vectorized OLS slope for y observed at x = [1, 2, 3].

    Analytical shortcuts for equally-spaced x:
    ┌─────────────────────────────┬─────────────────────┐
    │ Available quarters          │ Slope               │
    ├─────────────────────────────┼─────────────────────┤
    │ Q1 + Q2 + Q3  (all three)   │ (y3 − y1) / 2       │
    │ Q1       + Q3 (symmetric)   │ (y3 − y1) / 2       │
    │ Q1 + Q2                     │  y2 − y1            │
    │       Q2 + Q3               │  y3 − y2            │
    │ < 2 quarters                │  NaN                │
    └─────────────────────────────┴─────────────────────┘
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    y3 = np.asarray(y3, dtype=float)

    v1 = ~np.isnan(y1)
    v2 = ~np.isnan(y2)
    v3 = ~np.isnan(y3)

    slope = np.full(len(y1), np.nan, dtype=float)

    # Q1 + Q3 present (with or without Q2): symmetric, slope = (y3−y1)/2
    slope = np.where(v1 & v3, (y3 - y1) / 2.0, slope)

    # Q1 + Q2 only (no Q3): slope = y2−y1
    slope = np.where(v1 & v2 & ~v3, y2 - y1, slope)

    # Q2 + Q3 only (no Q1): slope = y3−y2
    slope = np.where(~v1 & v2 & v3, y3 - y2, slope)

    return slope


# ---------------------------------------------------------------------------
# Step 3: Targets
# ---------------------------------------------------------------------------

def _compute_targets(feat: pd.DataFrame) -> pd.DataFrame:
    """Q4 drop-off = Q4 per-minute rate minus Q1-Q3 per-minute rate."""
    feat["q4_pts_dropoff"]     = feat["pts_per_min_q4"]  - feat["pts_per_min_q1q3"]
    feat["q4_fg_pct_dropoff"]  = feat["fg_pct_q4"]       - feat["fg_pct_q1q3"]
    feat["q4_ast_dropoff"]     = feat["ast_per_min_q4"]  - feat["ast_per_min_q1q3"]
    feat["q4_to_surplus"]      = feat["to_per_min_q4"]   - feat["to_per_min_q1q3"]
    return feat


# ---------------------------------------------------------------------------
# Step 6: Team-level features
# ---------------------------------------------------------------------------

def _compute_team_features(q1q3: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team Q1-Q3 pace and score differential entering Q4.

    pace_q1q3             player's team possessions/48 in Q1-Q3
    game_pace             both teams' average pace in Q1-Q3
    score_diff_entering_q4  team pts − opponent pts through end of Q3
    """
    team = (
        q1q3.groupby(["GAME_ID", "TEAM_ID"])
        .agg(
            team_fga =("FGA",     "sum"),
            team_oreb=("OREB",    "sum"),
            team_to  =("TO",      "sum"),
            team_fta =("FTA",     "sum"),
            team_min =("MIN_DECIMAL", "sum"),   # sum over all players on team
            team_pts =("PTS",     "sum"),
        )
        .reset_index()
    )

    # Possession estimate: FGA − OREB + TO + 0.44·FTA
    team["possessions"] = (
        team["team_fga"]
        - team["team_oreb"]
        + team["team_to"]
        + 0.44 * team["team_fta"]
    )

    # Effective clock minutes = total player-minutes / 5 (five players on court)
    eff_min = (team["team_min"] / 5).clip(lower=1.0)
    team["pace_q1q3"] = team["possessions"] / eff_min * 48

    # Score differential: opponent pts = total game pts − own pts
    total_pts = team.groupby("GAME_ID")["team_pts"].transform("sum")
    team["opp_pts_q1q3"] = total_pts - team["team_pts"]
    team["score_diff_entering_q4"] = team["team_pts"] - team["opp_pts_q1q3"]

    # Game pace = mean of both teams
    game_pace = (
        team.groupby("GAME_ID")["pace_q1q3"]
        .mean()
        .reset_index()
        .rename(columns={"pace_q1q3": "game_pace"})
    )

    result = team[["GAME_ID", "TEAM_ID", "pace_q1q3", "score_diff_entering_q4"]]
    result = result.merge(game_pace, on="GAME_ID", how="left")
    return result


# ---------------------------------------------------------------------------
# Step 7: Season load
# ---------------------------------------------------------------------------

def _compute_season_load(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative season minutes and rolling average MPG entering each game.

    season_minutes_load        sum of full-game minutes in all PRIOR games this season
    minutes_per_game_season_avg  season_minutes_load / prior games played
    """
    # Full-game minutes per player-game (all 4 quarters)
    full_min = (
        raw_df.groupby(["GAME_ID", "PLAYER_ID"])["MIN_DECIMAL"]
        .sum()
        .reset_index()
        .rename(columns={"MIN_DECIMAL": "full_game_min"})
    )

    game_meta = (
        raw_df[["GAME_ID", "GAME_DATE", "SEASON"]]
        .drop_duplicates("GAME_ID")
    )
    full_min = full_min.merge(game_meta, on="GAME_ID", how="left")
    full_min = full_min.sort_values(
        ["PLAYER_ID", "SEASON", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    grp = full_min.groupby(["PLAYER_ID", "SEASON"])["full_game_min"]

    # Prior cumulative minutes = cumsum including current − current
    full_min["season_minutes_load"] = grp.transform(lambda x: x.cumsum() - x)

    # Number of games played BEFORE current game (0-indexed)
    full_min["prior_game_count"] = full_min.groupby(["PLAYER_ID", "SEASON"]).cumcount()

    # Average MPG over prior games (NaN for first game → imputed in _final_cleanup)
    full_min["minutes_per_game_season_avg"] = (
        full_min["season_minutes_load"]
        / full_min["prior_game_count"].replace(0, np.nan)
    )

    return full_min[["GAME_ID", "PLAYER_ID",
                     "season_minutes_load", "minutes_per_game_season_avg"]]


# ---------------------------------------------------------------------------
# Step 8: Player ages
# ---------------------------------------------------------------------------

def _get_player_ages(player_ids: list[int]) -> Optional[pd.DataFrame]:
    """Return a DataFrame of PLAYER_ID → birthdate, fetching from NBA API if needed.

    Caches to ``data/player_info.parquet``.  Incremental: only fetches players
    not already in the cache.
    """
    if PLAYER_INFO_PATH.exists():
        cached = pd.read_parquet(PLAYER_INFO_PATH)
        missing = [p for p in player_ids if p not in set(cached["PLAYER_ID"])]
        if not missing:
            return cached
        logger.info(f"Fetching info for {len(missing)} new players…")
        new_rows = _fetch_player_info_batch(missing)
        combined = pd.concat([cached, new_rows], ignore_index=True).drop_duplicates("PLAYER_ID")
        combined.to_parquet(PLAYER_INFO_PATH, index=False)
        return combined

    logger.info(
        f"Fetching birthdates for {len(player_ids)} players "
        f"(~{len(player_ids) * 0.6 / 60:.0f} min)…"
    )
    df = _fetch_player_info_batch(player_ids)
    df.to_parquet(PLAYER_INFO_PATH, index=False)
    return df


def _fetch_player_info_batch(player_ids: list[int]) -> pd.DataFrame:
    """Call CommonPlayerInfo for each player ID and return birthdate + position."""
    from nba_api.stats.endpoints import CommonPlayerInfo

    records = []
    for pid in tqdm(player_ids, desc="Player ages", unit="player", leave=False):
        try:
            ep = CommonPlayerInfo(player_id=int(pid))
            time.sleep(0.6)
            info = ep.common_player_info.get_data_frame()
            if not info.empty:
                row = info.iloc[0]
                records.append({
                    "PLAYER_ID": int(pid),
                    "birthdate": pd.to_datetime(row.get("BIRTHDATE", None)),
                    "position":  str(row.get("POSITION", "")),
                })
        except Exception as exc:
            logger.debug(f"Player {pid} info failed: {exc!r}")
            records.append({"PLAYER_ID": int(pid), "birthdate": pd.NaT, "position": ""})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Cleanup + imputation
# ---------------------------------------------------------------------------

def _final_cleanup(feat: pd.DataFrame) -> pd.DataFrame:
    """Impute missing feature values and drop rows with incomplete targets."""

    # minutes_per_game_season_avg: use player's own median, fallback to global
    global_mpg = feat["minutes_per_game_season_avg"].median()
    player_mpg = feat.groupby("PLAYER_ID")["minutes_per_game_season_avg"].transform("median")
    feat["minutes_per_game_season_avg"] = (
        feat["minutes_per_game_season_avg"]
        .fillna(player_mpg)
        .fillna(global_mpg)
    )

    # player_age: global median (~26 years for NBA)
    age_median = feat["player_age"].median()
    if pd.isna(age_median):
        age_median = 26.0
    feat["player_age"] = feat["player_age"].fillna(age_median)

    # rest_days: first game of season → assume 3 (typical opening-week gap)
    feat["rest_days"] = feat["rest_days"].fillna(3.0)

    # trend slopes: 0 = no detectable trend (neutral imputation)
    feat["usage_trend"]  = feat["usage_trend"].fillna(0.0)
    feat["fg_pct_trend"] = feat["fg_pct_trend"].fillna(0.0)

    # Drop rows where the core continuous targets are NaN
    # (q4_fg_pct_dropoff can be NaN for 0-FGA quarters — kept for other targets)
    feat = feat.dropna(subset=["q4_pts_dropoff", "q4_ast_dropoff", "q4_to_surplus"])

    return feat.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_summary(feat: pd.DataFrame) -> None:
    n        = len(feat)
    n_players = feat["PLAYER_ID"].nunique() if "PLAYER_ID" in feat.columns else "?"
    n_games   = feat["GAME_ID"].nunique()   if "GAME_ID"   in feat.columns else "?"
    seasons   = sorted(feat["SEASON"].unique()) if "SEASON" in feat.columns else []

    logger.info(
        f"Feature matrix: {n:,} rows | {n_players:,} players | "
        f"{n_games:,} games | seasons={seasons}"
    )

    if not all(c in feat.columns for c in TARGET_COLS):
        return

    logger.info("Target distributions:")
    for col in TARGET_COLS:
        if col in feat.columns:
            s = feat[col].dropna()
            logger.info(
                f"  {col:28s} mean={s.mean():+.3f}  std={s.std():.3f}  "
                f"[{s.min():.2f}, {s.max():.2f}]  nan={feat[col].isna().sum()}"
            )

    # Early hypothesis checks
    if "cumulative_minutes_q1q3" in feat.columns:
        med = feat["cumulative_minutes_q1q3"].median()
        hi  = feat.loc[feat["cumulative_minutes_q1q3"] >  med, "q4_pts_dropoff"].mean()
        lo  = feat.loc[feat["cumulative_minutes_q1q3"] <= med, "q4_pts_dropoff"].mean()
        verdict = "SUPPORTS" if hi < lo else "contradicts"
        logger.info(
            f"Hypothesis (high vs low minutes) — "
            f"pts/min drop-off: high={hi:+.3f}  low={lo:+.3f}  "
            f"diff={hi-lo:+.3f}  [{verdict} fatigue hypothesis]"
        )

    if "is_back_to_back" in feat.columns:
        b2b    = feat.loc[feat["is_back_to_back"] == 1.0, "q4_pts_dropoff"].mean()
        rested = feat.loc[feat["is_back_to_back"] == 0.0, "q4_pts_dropoff"].mean()
        verdict = "SUPPORTS" if b2b < rested else "contradicts"
        logger.info(
            f"Hypothesis (B2B vs rested) — "
            f"pts/min drop-off: B2B={b2b:+.3f}  rested={rested:+.3f}  "
            f"diff={b2b-rested:+.3f}  [{verdict} fatigue hypothesis]"
        )
