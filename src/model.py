"""Q4 fatigue prediction model (Phase 3).

Trains an XGBoost gradient boosting model to predict Q4 per-minute
stat drop-offs given the pre-Q4 fatigue feature vector.

Training split: 2022-23 and 2023-24 seasons (train + validation).
Holdout test:   2024-25 season (never touched during training).

If only one season is cached locally, falls back to an 80/20
chronological split and warns you.

Key outputs
-----------
- Feature importance ranking
- RMSE and R² on holdout set, compared against the naive zero-drop-off baseline
- Segment analysis: starters vs bench, young vs veteran, back-to-back vs rested

Usage
-----
>>> from src.model import train, predict_q4_dropoff
>>> results = train()
>>> predict_q4_dropoff("LeBron James", minutes_so_far=32, pace=98, rest_days=1)
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "data" / "fatigue_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "data" / "feature_matrix.parquet"

TRAIN_SEASONS = ["2022-23", "2023-24"]
TEST_SEASON   = "2024-25"

FEATURE_COLS: list[str] = [
    "cumulative_minutes_q1q3",
    "pace_q1q3",
    "usage_trend",
    "fg_pct_trend",
    "rest_days",
    "season_minutes_load",
    "game_pace",
    "score_diff_entering_q4",
    "is_home",
    "player_age",
    "minutes_per_game_season_avg",
]

TARGET_COLS: list[str] = [
    "q4_pts_dropoff",
    "q4_fg_pct_dropoff",
    "q4_ast_dropoff",
    "q4_to_surplus",
]

_XGB_PARAMS: dict = {
    "n_estimators":     500,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "seed":             42,
    "verbosity":        0,
    "n_jobs":           -1,
}

_EARLY_STOP  = 30    # patience rounds
_VAL_FRAC    = 0.15  # fraction of train set held out for early stopping


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def train(
    feature_matrix_path: Path | None = None,
    force_retrain: bool = False,
) -> dict[str, Any]:
    """Train the XGBoost fatigue model and evaluate on the holdout season.

    Parameters
    ----------
    feature_matrix_path:
        Path to the feature_matrix.parquet produced by Phase 2.
        Defaults to ``data/feature_matrix.parquet``.
    force_retrain:
        Re-train even if a saved model already exists.

    Returns
    -------
    dict with keys:
        models, metrics, feature_importance, segment_analysis,
        player_profiles, feature_medians, train_seasons, test_season,
        n_train, n_test, trained_at
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost not installed: pip install xgboost")

    if MODEL_PATH.exists() and not force_retrain:
        logger.info("Loading cached model from %s", MODEL_PATH)
        artifact = _load_artifact()
        _print_results(artifact["metrics"], artifact["feature_importance"], artifact["segment_analysis"])
        return artifact

    # ── Load feature matrix ────────────────────────────────────────────────
    fm_path = feature_matrix_path or FEATURES_PATH
    if not fm_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {fm_path}.\n"
            "Run first:  python predict.py --features"
        )

    feat = pd.read_parquet(fm_path)
    logger.info("Loaded feature matrix: %d rows", len(feat))

    # ── Season split ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feat_test = _split_by_season(feat)
    logger.info(
        "Train: %d rows | Test: %d rows",
        len(X_train), len(X_test),
    )

    # ── Train one model per target ─────────────────────────────────────────
    models: dict[str, Any] = {}
    for target in TARGET_COLS:
        models[target] = _train_single_target(X_train, y_train[target], target, xgb)

    # ── Evaluate ───────────────────────────────────────────────────────────
    metrics = _evaluate_all(models, X_test, y_test)

    # ── Feature importance ─────────────────────────────────────────────────
    importance = _feature_importance(models)

    # ── Segment analysis ───────────────────────────────────────────────────
    seg_df = _segment_analysis(models, X_test, y_test, feat_test)

    # ── Player profiles for single-prediction lookups ──────────────────────
    player_profiles = _build_player_profiles(feat)
    feature_medians = {c: float(feat[c].median()) for c in FEATURE_COLS if c in feat.columns}

    actual_train_seasons: list[str] = sorted(
        str(s) for s in feat.loc[~feat.index.isin(X_test.index), "SEASON"].unique()
    ) if "SEASON" in feat.columns else list(TRAIN_SEASONS)

    artifact = {
        "models":             models,
        "metrics":            metrics,
        "feature_importance": importance,
        "segment_analysis":   seg_df,
        "player_profiles":    player_profiles,
        "feature_medians":    feature_medians,
        "train_seasons":      actual_train_seasons,
        "test_season":        TEST_SEASON,
        "n_train":            len(X_train),
        "n_test":             len(X_test),
        "trained_at":         datetime.utcnow().isoformat(),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(artifact, fh)
    logger.info("Model saved -> %s", MODEL_PATH)

    _print_results(metrics, importance, seg_df)
    return artifact


def predict_q4_dropoff(
    player_name: str,
    minutes_so_far: float,
    pace: float,
    rest_days: int,
) -> dict[str, float]:
    """Print and return Q4 fatigue predictions for one player entering Q4.

    Parameters
    ----------
    player_name:
        Full player name, e.g. "LeBron James".
    minutes_so_far:
        Cumulative minutes played through end of Q3.
    pace:
        Game pace (possessions per 48).
    rest_days:
        Days since last game (0 = back-to-back).

    Returns
    -------
    dict mapping each target name to its predicted value.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run first:  python predict.py --train"
        )

    artifact = _load_artifact()
    models   = artifact["models"]
    profiles = artifact.get("player_profiles", {})
    medians  = artifact.get("feature_medians", {})

    # ── Build feature vector ───────────────────────────────────────────────
    # Look up player profile (case-insensitive partial match)
    player_key = _find_player(player_name, profiles)
    base = dict(medians)                       # global fallback
    if player_key:
        base.update(profiles[player_key])      # player-specific override
        logger.info("Using profile for '%s'", player_key)
    else:
        logger.warning(
            "Player '%s' not found in training data — using league medians.", player_name
        )

    # Override with explicitly provided context
    base["cumulative_minutes_q1q3"] = float(minutes_so_far)
    base["pace_q1q3"]               = float(pace)
    base["game_pace"]               = float(pace)
    base["rest_days"]               = float(rest_days)
    # score_diff_entering_q4 is live context — default to 0 (tied game)
    base.setdefault("score_diff_entering_q4", 0.0)

    fv = pd.DataFrame([{c: base.get(c, 0.0) for c in FEATURE_COLS}])

    # ── Predict ────────────────────────────────────────────────────────────
    preds: dict[str, float] = {}
    for target, model in models.items():
        preds[target] = float(model.predict(fv)[0])

    # ── Print ──────────────────────────────────────────────────────────────
    b2b = "back-to-back" if rest_days == 0 else f"{rest_days} rest day{'s' if rest_days != 1 else ''}"
    name_display = player_key or player_name

    print(f"\nQ4 Fatigue Prediction: {name_display}")
    print(f"Context: {minutes_so_far:.0f} Q1-Q3 min | pace {pace:.0f} | {b2b}")
    print()
    labels = {
        "q4_pts_dropoff":    ("pts/min drop-off",   "pts per minute in Q4 vs Q1-Q3"),
        "q4_fg_pct_dropoff": ("FG% drop-off",        "shooting efficiency change in Q4"),
        "q4_ast_dropoff":    ("ast/min drop-off",   "assists per minute change in Q4"),
        "q4_to_surplus":     ("TO/min surplus",     "turnovers per minute change in Q4"),
    }
    for target, (label, desc) in labels.items():
        print(f"  {label:<22} {preds[target]:+.3f}   ({desc})")

    n_train = artifact.get("n_train", "?")
    print(f"\nModel trained on {n_train:,} player-games from {artifact.get('train_seasons', [])}.")
    if not player_key:
        print("(No prior data found for this player — predictions use league averages.)")

    return preds


# ═══════════════════════════════════════════════════════════════════════════
# Season split
# ═══════════════════════════════════════════════════════════════════════════

def _split_by_season(
    feat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (X_train, X_test, y_train, y_test, feat_test).

    Prefers the canonical 2022-24 train / 2024-25 test split.
    Falls back to an 80/20 chronological split if only one season is present.
    """
    available = set(feat["SEASON"].unique()) if "SEASON" in feat.columns else set()

    has_train = any(s in available for s in TRAIN_SEASONS)
    has_test  = TEST_SEASON in available

    if has_train and has_test:
        train_mask = feat["SEASON"].isin(TRAIN_SEASONS)
        test_mask  = feat["SEASON"] == TEST_SEASON
    else:
        logger.warning(
            "Seasons in cache: %s  — expected %s + %s. "
            "Using 80/20 chronological split.",
            sorted(available), TRAIN_SEASONS, TEST_SEASON,
        )
        if "GAME_DATE" in feat.columns:
            feat = feat.sort_values("GAME_DATE").reset_index(drop=True)
        n         = len(feat)
        cut       = int(n * 0.80)
        train_mask = feat.index < cut
        test_mask  = feat.index >= cut

    X_train    = feat.loc[train_mask, FEATURE_COLS]
    y_train    = feat.loc[train_mask, TARGET_COLS]
    X_test     = feat.loc[test_mask,  FEATURE_COLS]
    y_test     = feat.loc[test_mask,  TARGET_COLS]
    feat_test  = feat.loc[test_mask]

    return X_train, X_test, y_train, y_test, feat_test


# ═══════════════════════════════════════════════════════════════════════════
# Model training
# ═══════════════════════════════════════════════════════════════════════════

def _train_single_target(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target: str,
    xgb,
) -> Any:
    """Train one XGBoost regressor with early stopping on the given target.

    Drops NaN rows in y_train (relevant for q4_fg_pct_dropoff).
    Uses the last _VAL_FRAC fraction of training data as the early-stopping
    validation set (chronological, not random, to avoid leakage).
    """
    mask   = y_train.notna()
    X_fit  = X_train.loc[mask]
    y_fit  = y_train.loc[mask]

    n       = len(X_fit)
    cut     = max(int(n * (1 - _VAL_FRAC)), n - 500)
    X_tr, X_val = X_fit.iloc[:cut], X_fit.iloc[cut:]
    y_tr, y_val = y_fit.iloc[:cut], y_fit.iloc[cut:]

    model = xgb.XGBRegressor(**_XGB_PARAMS, early_stopping_rounds=_EARLY_STOP)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best = model.best_iteration
    logger.info(
        "[%s] best_iteration=%d  val_rmse=%.4f",
        target, best,
        model.evals_result()["validation_0"]["rmse"][best],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_all(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute RMSE and R² for each target, plus naive baseline comparison."""
    results: dict[str, dict[str, float]] = {}

    for target in TARGET_COLS:
        mask  = y_test[target].notna()
        y_true = y_test.loc[mask, target]
        y_pred = pd.Series(
            models[target].predict(X_test.loc[mask]),
            index=y_true.index,
        )

        model_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        model_r2   = float(r2_score(y_true, y_pred))

        # Naive baseline: predict 0 drop-off (Q4 identical to Q1-Q3)
        baseline_rmse = float(np.sqrt(mean_squared_error(y_true, np.zeros(len(y_true)))))
        baseline_r2   = float(r2_score(y_true, np.zeros(len(y_true))))

        pct_improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0.0

        results[target] = {
            "rmse":             model_rmse,
            "r2":               model_r2,
            "baseline_rmse":    baseline_rmse,
            "baseline_r2":      baseline_r2,
            "pct_improvement":  pct_improvement,
            "n":                int(mask.sum()),
        }

    return results


def _naive_baseline(X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict[str, float]:
    """Naive baseline: predict Q4 stats == Q1-Q3 average (zero drop-off)."""
    out: dict[str, float] = {}
    for target in TARGET_COLS:
        mask   = y_test[target].notna()
        y_true = y_test.loc[mask, target]
        out[target] = float(np.sqrt(mean_squared_error(y_true, np.zeros(len(y_true)))))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Feature importance
# ═══════════════════════════════════════════════════════════════════════════

def _feature_importance(models: dict[str, Any]) -> pd.DataFrame:
    """Aggregate XGBoost gain-based importance across all four models.

    Returns a DataFrame ranked by mean importance, with one column per target.
    """
    frames = []
    for target, model in models.items():
        scores = model.get_booster().get_score(importance_type="gain")
        s = pd.Series(scores, name=target)
        frames.append(s)

    df = pd.concat(frames, axis=1).fillna(0)
    df["mean_gain"] = df.mean(axis=1)
    df = df.sort_values("mean_gain", ascending=False)
    df.index.name = "feature"
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Segment analysis
# ═══════════════════════════════════════════════════════════════════════════

def _segment_analysis(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    feat_test: pd.DataFrame,
) -> pd.DataFrame:
    """RMSE broken down by three player-archetype dimensions.

    Segments:
      role       — Starter (START_POSITION != '') vs Bench
      age_group  — Young (<25), Prime (25-31), Veteran (>=32)
      fatigue    — Back-to-back (rest_days == 0) vs Rested
    """
    rows = []

    def _rmse_segment(mask: pd.Series, target: str) -> float | None:
        valid = mask & y_test[target].notna()
        if valid.sum() < 5:
            return None
        y_true = y_test.loc[valid, target]
        y_pred = models[target].predict(X_test.loc[valid])
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # ── Role ──────────────────────────────────────────────────────────────
    if "START_POSITION" in feat_test.columns:
        starter_mask = feat_test["START_POSITION"].notna() & (feat_test["START_POSITION"] != "")
        for label, mask in [("Starter", starter_mask), ("Bench", ~starter_mask)]:
            row = {"segment": label, "n": int(mask.sum())}
            for t in TARGET_COLS:
                row[t] = _rmse_segment(mask, t)
            rows.append(row)

    # ── Age group ─────────────────────────────────────────────────────────
    if "player_age" in X_test.columns:
        age = X_test["player_age"]
        groups = [
            ("Young  (<25)",   age < 25),
            ("Prime  (25-31)", (age >= 25) & (age < 32)),
            ("Veteran (>=32)", age >= 32),
        ]
        for label, mask in groups:
            row = {"segment": label, "n": int(mask.sum())}
            for t in TARGET_COLS:
                row[t] = _rmse_segment(mask, t)
            rows.append(row)

    # ── Back-to-back vs rested ─────────────────────────────────────────────
    if "rest_days" in X_test.columns:
        b2b_mask    = X_test["rest_days"] == 0.0
        rested_mask = X_test["rest_days"] > 0.0
        for label, mask in [("Back-to-back", b2b_mask), ("Rested", rested_mask)]:
            row = {"segment": label, "n": int(mask.sum())}
            for t in TARGET_COLS:
                row[t] = _rmse_segment(mask, t)
            rows.append(row)

    return pd.DataFrame(rows).set_index("segment")


# ═══════════════════════════════════════════════════════════════════════════
# Player profiles
# ═══════════════════════════════════════════════════════════════════════════

def _build_player_profiles(feat: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute per-player median feature values from the full dataset.

    Stored as {player_name_lower: {feature: median_value}}.
    Used by predict_q4_dropoff() to fill in features not provided by the user.
    """
    if "PLAYER_NAME" not in feat.columns:
        return {}

    profiles: dict[str, dict[str, float]] = {}
    for name, grp in feat.groupby("PLAYER_NAME"):
        medians = {}
        for col in FEATURE_COLS:
            if col in grp.columns:
                val = grp[col].median()
                if not np.isnan(val):
                    medians[col] = float(val)
        profiles[str(name).lower()] = medians

    return profiles


def _find_player(player_name: str, profiles: dict[str, dict]) -> str | None:
    """Case-insensitive partial match of player_name against profile keys."""
    query = player_name.lower().strip()
    if query in profiles:
        return query
    # Partial match: prefer the longest key that contains the query
    matches = [k for k in profiles if query in k or k in query]
    if matches:
        return max(matches, key=len)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Persistence helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_artifact() -> dict[str, Any]:
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


# ═══════════════════════════════════════════════════════════════════════════
# Results printer
# ═══════════════════════════════════════════════════════════════════════════

def _print_results(
    metrics: dict[str, dict[str, float]],
    importance: pd.DataFrame,
    seg_df: pd.DataFrame,
) -> None:
    _sep = "=" * 64

    # ── Metrics table ──────────────────────────────────────────────────────
    print(f"\n{_sep}")
    print("HOLDOUT METRICS  (model vs naive zero-drop-off baseline)")
    print(_sep)
    print(f"  {'Target':<26} {'RMSE':>7}  {'R2':>6}  {'Baseline':>9}  {'Imprv':>6}  {'N':>5}")
    print(f"  {'-'*26} {'-'*7}  {'-'*6}  {'-'*9}  {'-'*6}  {'-'*5}")
    for target, m in metrics.items():
        print(
            f"  {target:<26} {m['rmse']:>7.4f}  {m['r2']:>+6.3f}  "
            f"{m['baseline_rmse']:>9.4f}  {m['pct_improvement']:>+5.1f}%  {m['n']:>5,}"
        )

    # ── Feature importance ─────────────────────────────────────────────────
    print(f"\n{_sep}")
    print("FEATURE IMPORTANCE  (XGBoost gain, mean across targets)")
    print(_sep)
    top = importance.head(len(FEATURE_COLS))
    max_gain = top["mean_gain"].max() if top["mean_gain"].max() > 0 else 1
    for feat_name, row in top.iterrows():
        bar_len = int(row["mean_gain"] / max_gain * 30)
        bar = "#" * bar_len
        print(f"  {feat_name:<34} {row['mean_gain']:>9.1f}  {bar}")

    # ── Segment analysis ───────────────────────────────────────────────────
    if not seg_df.empty:
        print(f"\n{_sep}")
        print("SEGMENT ANALYSIS  (RMSE by player archetype)")
        print(_sep)
        cols_to_show = [c for c in TARGET_COLS if c in seg_df.columns]
        header = f"  {'Segment':<22} {'N':>5}  " + "  ".join(f"{c.split('_')[1].upper():>8}" for c in cols_to_show)
        print(header)
        print(f"  {'-'*22} {'-'*5}  " + "  ".join("-" * 8 for _ in cols_to_show))
        for seg, row in seg_df.iterrows():
            vals = "  ".join(
                f"{row[c]:>8.4f}" if pd.notna(row.get(c)) else f"{'---':>8}"
                for c in cols_to_show
            )
            print(f"  {str(seg):<22} {int(row['n']):>5}  {vals}")

    print(f"\n{_sep}\n")
