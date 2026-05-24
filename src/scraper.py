"""NBA quarter-by-quarter data scraper (Phase 1).

Collects per-quarter box scores for all regular season games across the
specified seasons using the nba_api package, caching results to Parquet so
the 2–3 hour initial scrape only runs once.

Output layout
-------------
data/
  game_ids/
    game_ids_2022-23.parquet      # one row per game, schedule context
    game_ids_2023-24.parquet
    game_ids_2024-25.parquet
  box_scores/
    {game_id}.parquet             # Q1–Q4 player stats for one game
  player_game_quarters_2022-23.parquet   # assembled season dataset
  player_game_quarters_2023-24.parquet
  player_game_quarters_2024-25.parquet
  player_game_quarters_all.parquet       # full three-season dataset

Usage
-----
>>> from src.scraper import build_dataset
>>> df = build_dataset(seasons=["2022-23", "2023-24", "2024-25"])
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from nba_api.stats.endpoints import BoxScoreTraditionalV2, LeagueGameFinder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GAME_IDS_DIR = DATA_DIR / "game_ids"
BOX_SCORES_DIR = DATA_DIR / "box_scores"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RATE_LIMIT_SLEEP: float = 0.6   # seconds between NBA API calls
RETRY_DELAYS: list[float] = [2.0, 5.0, 15.0]  # per-attempt backoff

DEFAULT_SEASONS: list[str] = ["2022-23", "2023-24", "2024-25"]
QUARTERS: list[int] = [1, 2, 3, 4]

# range_type=2 (period + time range combined) is the only mode that correctly
# returns per-quarter stats. Strategies 0, 1 silently return full-game totals.
# Time boundaries are in tenths of a second (12 min × 60 s × 10 = 7200 per quarter).
QUARTER_TIME_RANGES: dict[int, tuple[int, int]] = {
    1: (0,     7200),
    2: (7201,  14400),   # +1 on start: exclude Q1's last-second shot from Q2
    3: (14401, 21600),   # same for Q2/Q3 boundary
    4: (21601, 28800),   # same for Q3/Q4 boundary
}

# Columns expected from BoxScoreTraditionalV2 player_stats
BOX_SCORE_NUMERIC_COLS: list[str] = [
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TO", "PF", "PTS", "PLUS_MINUS",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_minutes(min_str) -> float:
    """Convert NBA API "MM:SS" string to decimal minutes."""
    if min_str is None or (isinstance(min_str, float) and np.isnan(min_str)):
        return 0.0
    s = str(min_str).strip()
    if not s:
        return 0.0
    if ":" in s:
        try:
            m, sec = s.split(":", 1)
            return float(m) + float(sec) / 60.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _call_api(endpoint_cls, **kwargs):
    """
    Instantiate an nba_api endpoint with rate-limiting and exponential
    backoff on failure.

    Raises RuntimeError after all retry attempts are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt, backoff in enumerate(RETRY_DELAYS, 1):
        try:
            result = endpoint_cls(**kwargs)
            time.sleep(RATE_LIMIT_SLEEP)
            return result
        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"{endpoint_cls.__name__} attempt {attempt}/{len(RETRY_DELAYS)} "
                f"failed: {exc!r}. Retrying in {backoff}s…"
            )
            time.sleep(backoff)
    raise RuntimeError(
        f"{endpoint_cls.__name__} failed after {len(RETRY_DELAYS)} attempts"
    ) from last_exc


# ---------------------------------------------------------------------------
# Game IDs
# ---------------------------------------------------------------------------

def fetch_game_ids(season: str, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch all regular-season game IDs and schedule context for one season.

    Caches results to ``data/game_ids/game_ids_{season}.parquet``.

    Parameters
    ----------
    season:
        NBA season string, e.g. ``"2022-23"``.
    force_refresh:
        Re-fetch from the API even if a cache file already exists.

    Returns
    -------
    DataFrame with columns:
        GAME_ID, GAME_DATE, SEASON,
        HOME_TEAM_ID, HOME_TEAM_ABB,
        AWAY_TEAM_ID, AWAY_TEAM_ABB,
        REST_DAYS_HOME, REST_DAYS_AWAY
    """
    cache_path = GAME_IDS_DIR / f"game_ids_{season}.parquet"

    if cache_path.exists() and not force_refresh:
        logger.info(f"[{season}] Loading cached game IDs → {cache_path.name}")
        return pd.read_parquet(cache_path)

    logger.info(f"[{season}] Fetching game schedule from NBA API…")

    endpoint = _call_api(
        LeagueGameFinder,
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    raw: pd.DataFrame = endpoint.get_data_frames()[0]

    if raw.empty:
        logger.warning(f"[{season}] LeagueGameFinder returned no data.")
        return pd.DataFrame()

    # Each game appears twice: home team row has "vs." in MATCHUP, away has "@"
    raw["IS_HOME"] = raw["MATCHUP"].str.contains(r"vs\.", regex=True)
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])

    home = (
        raw[raw["IS_HOME"]][["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION"]]
        .rename(columns={"TEAM_ID": "HOME_TEAM_ID", "TEAM_ABBREVIATION": "HOME_TEAM_ABB"})
        .copy()
    )
    away = (
        raw[~raw["IS_HOME"]][["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION"]]
        .rename(columns={"TEAM_ID": "AWAY_TEAM_ID", "TEAM_ABBREVIATION": "AWAY_TEAM_ABB"})
        .copy()
    )

    games = home.merge(away, on="GAME_ID", how="inner")
    games["SEASON"] = season
    games = _compute_rest_days(games)

    GAME_IDS_DIR.mkdir(parents=True, exist_ok=True)
    games.to_parquet(cache_path, index=False)

    logger.info(f"[{season}] {len(games)} games saved → {cache_path.name}")
    return games


def _compute_rest_days(games: pd.DataFrame) -> pd.DataFrame:
    """Add REST_DAYS_HOME and REST_DAYS_AWAY columns.

    Rest days = calendar gap between consecutive games - 1.
    A back-to-back is REST_DAYS == 0; NaN means the team's first game.
    """
    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    # Build a flat team-game timeline (one row per team per game)
    home_tl = games[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID"]].copy()
    home_tl.rename(columns={"HOME_TEAM_ID": "TEAM_ID"}, inplace=True)
    home_tl["_is_home"] = True

    away_tl = games[["GAME_ID", "GAME_DATE", "AWAY_TEAM_ID"]].copy()
    away_tl.rename(columns={"AWAY_TEAM_ID": "TEAM_ID"}, inplace=True)
    away_tl["_is_home"] = False

    timeline = pd.concat([home_tl, away_tl], ignore_index=True).sort_values(
        ["TEAM_ID", "GAME_DATE"]
    )

    timeline["_prev_date"] = timeline.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    timeline["REST_DAYS"] = (timeline["GAME_DATE"] - timeline["_prev_date"]).dt.days - 1

    home_rest = (
        timeline[timeline["_is_home"]][["GAME_ID", "REST_DAYS"]]
        .rename(columns={"REST_DAYS": "REST_DAYS_HOME"})
    )
    away_rest = (
        timeline[~timeline["_is_home"]][["GAME_ID", "REST_DAYS"]]
        .rename(columns={"REST_DAYS": "REST_DAYS_AWAY"})
    )

    games = games.merge(home_rest, on="GAME_ID", how="left")
    games = games.merge(away_rest, on="GAME_ID", how="left")
    return games


# ---------------------------------------------------------------------------
# Per-game box scores
# ---------------------------------------------------------------------------

def _is_game_cached(game_id: str) -> bool:
    """True if a non-empty box score Parquet already exists for this game."""
    p = BOX_SCORES_DIR / f"{game_id}.parquet"
    return p.exists() and p.stat().st_size > 0


def fetch_box_score(game_id: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Fetch per-quarter player stats for a single game (Q1–Q4 only).

    Makes four API calls — one per quarter — and caches all four periods
    together in ``data/box_scores/{game_id}.parquet``.

    Parameters
    ----------
    game_id:
        10-digit NBA game identifier, e.g. ``"0022200001"``.
    force_refresh:
        Re-fetch even if a cache file exists.

    Returns
    -------
    DataFrame with a ``PERIOD`` column (1–4) and all standard box score
    columns, or ``None`` if every quarter fetch failed.
    """
    cache_path = BOX_SCORES_DIR / f"{game_id}.parquet"

    if _is_game_cached(game_id) and not force_refresh:
        return pd.read_parquet(cache_path)

    quarter_frames: list[pd.DataFrame] = []

    for period in QUARTERS:
        start_range, end_range = QUARTER_TIME_RANGES[period]
        try:
            endpoint = _call_api(
                BoxScoreTraditionalV2,
                game_id=game_id,
                start_period=period,
                end_period=period,
                start_range=start_range,
                end_range=end_range,
                range_type=2,   # combined period + time range (only mode that filters correctly)
            )
            stats: pd.DataFrame = endpoint.player_stats.get_data_frame()

            if stats.empty:
                logger.debug(f"[{game_id}] Q{period}: empty response")
                continue

            stats = stats.copy()
            stats["PERIOD"] = period
            stats["MIN_DECIMAL"] = stats["MIN"].apply(parse_minutes)
            quarter_frames.append(stats)

        except Exception as exc:
            logger.debug(f"[{game_id}] Q{period} fetch error: {exc!r}")

    if not quarter_frames:
        logger.warning(f"[{game_id}] No quarter data retrieved — skipping.")
        return None

    df = pd.concat(quarter_frames, ignore_index=True)

    BOX_SCORES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Season assembly
# ---------------------------------------------------------------------------

def scrape_season(season: str, force_refresh: bool = False) -> pd.DataFrame:
    """Scrape and assemble per-quarter player stats for one season.

    Steps
    -----
    1. Fetch game IDs + schedule (cached after first run).
    2. Fetch box scores per game (cached per game_id — safe to interrupt).
    3. Merge game context: dates, rest days, home/away flag.

    The season-level DataFrame is saved to
    ``data/player_game_quarters_{season}.parquet``.

    Parameters
    ----------
    season:
        NBA season string, e.g. ``"2023-24"``.
    force_refresh:
        Re-scrape everything even if the season cache exists. Per-game
        Parquet files are also re-fetched when this is True.

    Returns
    -------
    DataFrame with ~25K–35K rows (players × games × quarters).
    """
    output_path = DATA_DIR / f"player_game_quarters_{season}.parquet"

    if output_path.exists() and not force_refresh:
        logger.info(f"[{season}] Loading cached season dataset.")
        return pd.read_parquet(output_path)

    games = fetch_game_ids(season, force_refresh=force_refresh)
    if games.empty:
        return pd.DataFrame()

    game_ids = games["GAME_ID"].unique().tolist()
    api_calls_est = len(game_ids) * len(QUARTERS) * RATE_LIMIT_SLEEP
    logger.info(
        f"[{season}] Scraping {len(game_ids)} games "
        f"(≈{api_calls_est / 60:.0f} min for fresh data, "
        f"cached games are instant)…"
    )

    box_frames: list[pd.DataFrame] = []
    n_failed = 0

    with tqdm(game_ids, desc=season, unit="game", dynamic_ncols=True) as pbar:
        for game_id in pbar:
            result = fetch_box_score(game_id, force_refresh=force_refresh)
            if result is not None:
                box_frames.append(result)
            else:
                n_failed += 1
            pbar.set_postfix({"ok": len(box_frames), "fail": n_failed})

    if not box_frames:
        logger.error(f"[{season}] No data collected — aborting.")
        return pd.DataFrame()

    df = pd.concat(box_frames, ignore_index=True)
    df = _merge_game_context(df, games)
    df = _clean_dtypes(df)

    df.to_parquet(output_path, index=False)

    n_players = df["PLAYER_ID"].nunique()
    n_games = df["GAME_ID"].nunique()
    logger.info(
        f"[{season}] Saved {len(df):,} rows "
        f"({n_players:,} players, {n_games:,} games) → {output_path.name}"
    )
    if n_failed:
        logger.warning(f"[{season}] {n_failed}/{len(game_ids)} games failed.")

    return df


def _merge_game_context(df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Attach game-level context columns to the player-quarter DataFrame."""
    context = games[[
        "GAME_ID", "GAME_DATE", "SEASON",
        "HOME_TEAM_ID", "HOME_TEAM_ABB",
        "AWAY_TEAM_ID", "AWAY_TEAM_ABB",
        "REST_DAYS_HOME", "REST_DAYS_AWAY",
    ]]

    df = df.merge(context, on="GAME_ID", how="left")

    df["IS_HOME"] = df["TEAM_ID"] == df["HOME_TEAM_ID"]
    df["REST_DAYS"] = np.where(
        df["IS_HOME"], df["REST_DAYS_HOME"], df["REST_DAYS_AWAY"]
    )
    df["IS_BACK_TO_BACK"] = df["REST_DAYS"] == 0

    return df


def _clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all box-score stat columns to float; parse GAME_DATE."""
    for col in BOX_SCORE_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    return df


# ---------------------------------------------------------------------------
# Dataset builder (main entry point)
# ---------------------------------------------------------------------------

def build_dataset(
    seasons: list[str] = DEFAULT_SEASONS,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build the complete player-game-quarter dataset across all seasons.

    This is the Phase 1 entry point. On the first run it scrapes ~3,900
    games (~14,760 API calls at 0.6 s each ≈ 2.5 hours). Subsequent calls
    load from cache instantly.

    Parameters
    ----------
    seasons:
        NBA seasons to collect, e.g. ``["2022-23", "2023-24", "2024-25"]``.
    force_refresh:
        Re-fetch everything from the API. Use if the data looks stale.

    Returns
    -------
    DataFrame with ~97K rows:
        GAME_ID, PERIOD, SEASON, GAME_DATE,
        PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, START_POSITION,
        MIN, MIN_DECIMAL, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT,
        FTM, FTA, FT_PCT, OREB, DREB, REB, AST, STL, BLK, TO, PF, PTS,
        PLUS_MINUS, IS_HOME, REST_DAYS, IS_BACK_TO_BACK,
        HOME_TEAM_ID, HOME_TEAM_ABB, AWAY_TEAM_ID, AWAY_TEAM_ABB,
        REST_DAYS_HOME, REST_DAYS_AWAY
    """
    output_path = DATA_DIR / "player_game_quarters_all.parquet"

    if output_path.exists() and not force_refresh:
        logger.info(f"Loading full cached dataset from {output_path.name}")
        df = pd.read_parquet(output_path)
        _log_dataset_summary(df)
        return df

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    season_frames: list[pd.DataFrame] = []
    for season in seasons:
        season_df = scrape_season(season, force_refresh=force_refresh)
        if not season_df.empty:
            season_frames.append(season_df)

    if not season_frames:
        logger.error("No data collected for any season.")
        return pd.DataFrame()

    full_df = pd.concat(season_frames, ignore_index=True)
    full_df.to_parquet(output_path, index=False)

    _log_dataset_summary(full_df)
    return full_df


def _log_dataset_summary(df: pd.DataFrame) -> None:
    seasons = sorted(df["SEASON"].unique()) if "SEASON" in df.columns else ["?"]
    n_players = df["PLAYER_ID"].nunique() if "PLAYER_ID" in df.columns else "?"
    n_games = df["GAME_ID"].nunique() if "GAME_ID" in df.columns else "?"
    logger.info(
        f"Dataset summary: {len(df):,} rows | "
        f"{n_players:,} players | {n_games:,} games | "
        f"seasons={seasons}"
    )
