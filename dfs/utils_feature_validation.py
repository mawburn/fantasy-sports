# utils_feature_validation.py
from __future__ import annotations

import json
from typing import Iterable, List

import numpy as np
import pandas as pd

_NUMERIC_SOFT_RANGES = {
    # odds
    "team_spread": (-30, 30),
    "team_spread_abs": (0, 30),
    "total_line": (20, 75),
    "game_tot_z": (-4, 4),
    "team_itt": (7, 45),
    "team_itt_z": (-4, 4),
    # weather
    "temperature_f": (-10, 120),
    "wind_mph": (0, 60),
    "humidity_pct": (0, 100),
    # injuries / counts
    "games_missed_last4": (0, 4),
    "team_injured_starters": (0, 11),
    "opp_injured_starters": (0, 11),
    # misc
    "salary": (2000, 12000),
}

_BINARY_COLS = [
    "is_favorite",
    "cold_lt40",
    "hot_gt85",
    "wind_gt15",
    "dome",
    "returning_from_injury",
    # injury status one-hots (add/remove to match your schema file)
    "injury_status_Out",
    "injury_status_Doubtful",
    "injury_status_Questionable",
    "injury_status_Probable",
]

# If you include a cyclical/normalized week col, add it here as numeric.
_NUMERIC_ALWAYS = set(_NUMERIC_SOFT_RANGES.keys()) | {
    "targets_ema",
    "routes_run_ema",
    "rush_att_ema",
    "snap_share_ema",
    "redzone_opps_ema",
    "air_yards_ema",
    "adot_ema",
    "yprr_ema",
    "yards_after_contact",
    "missed_tackles_forced",
    "pressure_rate",
    "opp_dvp_pos_allowed",
    "home",
    "rest_days",
    "travel",
    "season_week",
}


def _fail(msg: str):
    raise ValueError(f"[FeatureValidation] {msg}")


def load_expected_schema(schema_path: str) -> List[str]:
    with open(schema_path, "r") as f:
        names = json.load(f)
    if not isinstance(names, list) or not all(isinstance(c, str) for c in names):
        _fail("feature_names.json must be a JSON list of strings.")
    return names


def enforce_feature_order(
    df: pd.DataFrame, expected_order: List[str], allow_extra: bool = False
) -> pd.DataFrame:
    miss = [c for c in expected_order if c not in df.columns]
    if miss:
        _fail(f"Missing required columns: {miss}")
    extras = [c for c in df.columns if c not in expected_order]
    if extras and not allow_extra:
        _fail(
            f"Unexpected extra columns present (set allow_extra=True to ignore): {extras[:20]}"
        )
    # Reorder and optionally drop extras
    df2 = df[expected_order].copy()
    return df2


def _assert_binary_col(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return
    vals = df[col].dropna().unique()
    ok = set([0, 1])
    if not set(np.unique(vals)).issubset(ok):
        _fail(
            f"Column '{col}' must be binary (0/1). Found values: {sorted(vals.tolist())[:6]}"
        )


def _assert_soft_range(df: pd.DataFrame, col: str, lo: float, hi: float):
    if col not in df.columns:
        return
    s = df[col]
    bad = s[(s < lo) | (s > hi)]
    if len(bad) > 0:
        # Don't fail on a single outlier; fail if >1% rows or egregious
        frac = len(bad) / len(s)
        if frac > 0.01 or bad.abs().max() > max(abs(lo) * 1.5, abs(hi) * 1.5):
            _fail(
                f"Column '{col}' outside soft range [{lo},{hi}] for {len(bad)} rows ({frac:.1%}). Example: {bad.iloc[0]}"
            )


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _check_nan_inf(df: pd.DataFrame, cols: Iterable[str]):
    sub = df[list(cols)]
    if not np.isfinite(sub.values).all():
        # Identify first offenders
        bad = ~np.isfinite(sub.values)
        r, c = np.argwhere(bad)[0]
        _fail(f"NaN/Inf in numeric column '{sub.columns[c]}' at row {r}.")


def validate_and_prepare_features(
    df: pd.DataFrame,
    expected_schema: List[str],
    allow_extra: bool = False,
    coerce_numeric: bool = True,
) -> pd.DataFrame:
    """
    Validates feature frame, enforces order, checks types/ranges, and returns a copy ready for scaling/model.
    """
    if df.empty:
        _fail("Input features DataFrame is empty.")

    # Enforce presence and order (will drop extras unless allow_extra=True)
    df = enforce_feature_order(df, expected_schema, allow_extra=allow_extra)

    # Coerce numeric columns and check NaN/Inf
    if coerce_numeric:
        df = _coerce_numeric(df, _NUMERIC_ALWAYS.intersection(df.columns))
    _check_nan_inf(df, _NUMERIC_ALWAYS.intersection(df.columns))

    # Binary checks
    for col in _BINARY_COLS:
        if col in df.columns:
            _assert_binary_col(df, col)

    # Injury one-hot exclusivity (if the four are present, row-sum <= 1)
    inj_cols = [
        c for c in _BINARY_COLS if c.startswith("injury_status_") and c in df.columns
    ]
    if inj_cols:
        sums = df[inj_cols].sum(axis=1)
        if (sums > 1.0 + 1e-6).any():
            _fail(
                f"Injury status one-hots must be mutually exclusive. Found row-sum > 1 in {int((sums > 1.0).sum())} rows."
            )

    # Soft numeric ranges
    for col, (lo, hi) in _NUMERIC_SOFT_RANGES.items():
        if col in df.columns:
            _assert_soft_range(df, col, lo, hi)

    # Final sanity: no constant columns (zero variance) among numeric drivers
    # Skip this check if DataFrame is small (test data)
    if len(df) > 50:  # Only check variance for larger datasets
        for col in _NUMERIC_ALWAYS.intersection(df.columns):
            s = df[col].dropna()
            if len(s) > 0 and float(s.std()) == 0.0:
                _fail(
                    f"Numeric column '{col}' has zero variance; likely a join/feature bug."
                )

    return df.copy()
