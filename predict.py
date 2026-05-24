#!/usr/bin/env python3
"""NBA Fourth Quarter Fatigue Predictor — CLI entry point.

Examples
--------
# Step 1: Scrape data (run once — takes 2-3 hours, then cached)
python predict.py --scrape --seasons 2022-23 2023-24 2024-25

# Step 2: Predict Q4 fatigue for a specific player entering a game
python predict.py --player "LeBron James" --minutes-so-far 32 --pace 98 --rest-days 1

# Step 3: Run full backtest on 2024-25 season
python predict.py --backtest --season 2024-25

# Step 4: Generate all visualizations into figures/
python predict.py --visualize

# Force re-scrape (ignores cache)
python predict.py --scrape --seasons 2024-25 --force-refresh
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="NBA Fourth Quarter Fatigue Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Mode (mutually exclusive) ──────────────────────────────────────────
    modes = parser.add_argument_group("mode (pick one)")
    modes.add_argument(
        "--scrape",
        action="store_true",
        help="Scrape NBA API data and build the base dataset.",
    )
    modes.add_argument(
        "--features",
        action="store_true",
        help="Build the feature matrix from scraped data (Phase 2).",
    )
    modes.add_argument(
        "--train",
        action="store_true",
        help="Train the XGBoost fatigue model (Phase 3).",
    )
    modes.add_argument(
        "--backtest",
        action="store_true",
        help="Run game-by-game Q4 prediction backtest.",
    )
    modes.add_argument(
        "--visualize",
        action="store_true",
        help="Generate all diagnostic plots into figures/.",
    )
    modes.add_argument(
        "--player",
        type=str,
        metavar="NAME",
        help="Predict Q4 fatigue for a specific player (requires --minutes-so-far, --pace, --rest-days).",
    )

    # ── General options ────────────────────────────────────────────────────
    general = parser.add_argument_group("general options")
    general.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass cache and rebuild from scratch (works with --scrape, --features, --train).",
    )

    # ── Scrape options ─────────────────────────────────────────────────────
    scrape = parser.add_argument_group("scrape options")
    scrape.add_argument(
        "--seasons",
        nargs="+",
        default=["2022-23", "2023-24", "2024-25"],
        metavar="YYYY-YY",
        help="Season(s) to scrape (default: 2022-23 2023-24 2024-25).",
    )

    # ── Backtest options ───────────────────────────────────────────────────
    bt = parser.add_argument_group("backtest options")
    bt.add_argument(
        "--season",
        type=str,
        default="2024-25",
        metavar="YYYY-YY",
        help="Season to backtest (default: 2024-25).",
    )

    # ── Player prediction options ──────────────────────────────────────────
    pred = parser.add_argument_group("player prediction options")
    pred.add_argument(
        "--minutes-so-far",
        type=float,
        metavar="MIN",
        help="Cumulative minutes played through end of Q3.",
    )
    pred.add_argument(
        "--pace",
        type=float,
        metavar="PACE",
        help="Game pace (possessions per 48).",
    )
    pred.add_argument(
        "--rest-days",
        type=int,
        metavar="N",
        help="Days since last game (0 = back-to-back).",
    )

    return parser


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_scrape(seasons: list[str], force_refresh: bool) -> None:
    from src.scraper import build_dataset

    logger.info(f"Starting data collection for: {seasons}")
    df = build_dataset(seasons=seasons, force_refresh=force_refresh)

    if df.empty:
        logger.error("Scraping returned no data.")
        sys.exit(1)

    print(
        f"\nDone. Dataset: {len(df):,} rows × {df.shape[1]} cols\n"
        f"  Players : {df['PLAYER_ID'].nunique():,}\n"
        f"  Games   : {df['GAME_ID'].nunique():,}\n"
        f"  Seasons : {sorted(df['SEASON'].unique())}\n"
        f"  Saved   → data/player_game_quarters_all.parquet"
    )


def run_train(force_retrain: bool) -> None:
    from src.model import train
    train(force_retrain=force_retrain)


def run_features(force_refresh: bool) -> None:
    from src.features import build_feature_matrix

    X, y = build_feature_matrix(force_refresh=force_refresh)
    print(
        f"\nFeature matrix built: {len(X):,} player-games\n"
        f"  X shape : {X.shape}\n"
        f"  y shape : {y.shape}\n"
        f"  Saved   → data/feature_matrix.parquet"
    )


def run_backtest(season: str) -> None:
    try:
        from src.evaluate import run_backtest as _bt
    except ImportError:
        logger.error("Evaluation module not yet implemented (Phase 4).")
        sys.exit(1)

    _bt(season=season)


def run_visualize() -> None:
    try:
        from src.visualize import generate_all
    except ImportError:
        logger.error("Visualization module not yet implemented (Phase 5).")
        sys.exit(1)

    generate_all()
    print("Plots saved to figures/")


def run_player_prediction(
    player_name: str,
    minutes_so_far: float,
    pace: float,
    rest_days: int,
) -> None:
    try:
        from src.model import predict_q4_dropoff
    except ImportError:
        logger.error("Model module not yet implemented (Phase 3).")
        sys.exit(1)

    predict_q4_dropoff(
        player_name=player_name,
        minutes_so_far=minutes_so_far,
        pace=pace,
        rest_days=rest_days,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.scrape:
        run_scrape(args.seasons, args.force_refresh)

    elif args.train:
        run_train(args.force_refresh)

    elif args.features:
        run_features(args.force_refresh)

    elif args.backtest:
        run_backtest(args.season)

    elif args.visualize:
        run_visualize()

    elif args.player:
        required = {
            "--minutes-so-far": args.minutes_so_far,
            "--pace": args.pace,
            "--rest-days": args.rest_days,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            parser.error(f"--player requires: {', '.join(missing)}")
        run_player_prediction(
            player_name=args.player,
            minutes_so_far=args.minutes_so_far,
            pace=args.pace,
            rest_days=args.rest_days,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
