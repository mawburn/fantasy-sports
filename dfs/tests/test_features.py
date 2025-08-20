# tests/test_features.py
import pandas as pd
import numpy as np
import json
from utils_feature_validation import validate_and_prepare_features

def _schema():
    return [
        # odds
        "team_spread","team_spread_abs","total_line","game_tot_z","team_itt","team_itt_z","is_favorite",
        # weather
        "temperature_f","wind_mph","humidity_pct","cold_lt40","hot_gt85","wind_gt15","dome",
        # injuries
        "injury_status_Out","injury_status_Doubtful","injury_status_Questionable","injury_status_Probable",
        "games_missed_last4","practice_trend","returning_from_injury","team_injured_starters","opp_injured_starters",
        # usage (examples)
        "targets_ema","routes_run_ema","rush_att_ema","snap_share_ema","redzone_opps_ema",
        "air_yards_ema","adot_ema","yprr_ema",
        # efficiency
        "yards_after_contact","missed_tackles_forced","pressure_rate","opp_dvp_pos_allowed",
        # context
        "salary","home","rest_days","travel","season_week"
    ]

def test_ok_frame_passes():
    exp = _schema()
    n = 10
    df = pd.DataFrame({c: [0] * n for c in exp})
    # Fill with plausible values
    df["team_spread"] = np.linspace(-7, 7, n)
    df["team_spread_abs"] = df["team_spread"].abs()
    df["total_line"] = np.random.uniform(42, 55, n)
    df["game_tot_z"] = np.random.normal(0, 1, n)
    df["team_itt"] = np.random.uniform(20, 30, n)
    df["team_itt_z"] = np.random.normal(0, 1, n)
    df["is_favorite"] = (df["team_spread"] < 0).astype(int)
    df["temperature_f"] = np.random.uniform(30, 90, n)
    df["wind_mph"] = np.random.uniform(0, 20, n)
    df["humidity_pct"] = np.random.uniform(30, 90, n)
    for b in ["cold_lt40","hot_gt85","wind_gt15","dome","returning_from_injury","home"]:
        df[b] = 0
    # one-hot injuries
    df["injury_status_Out"] = 0
    df["injury_status_Doubtful"] = 0
    df["injury_status_Questionable"] = 1
    df["injury_status_Probable"] = 0
    df["games_missed_last4"] = np.random.choice([0, 1, 2], n)
    df["practice_trend"] = np.random.choice([-1, 0, 1], n)
    df["team_injured_starters"] = np.random.choice([0, 1, 2], n)
    df["opp_injured_starters"] = np.random.choice([0, 1, 2], n)
    # numerics - add variation to avoid zero variance
    df["salary"] = np.random.uniform(5000, 8000, n)
    df["rest_days"] = np.random.choice([6, 7, 8], n)
    df["travel"] = np.random.uniform(0, 1000, n)
    df["season_week"] = np.random.choice([1, 2, 3, 4, 5], n)
    for c in ["targets_ema","routes_run_ema","rush_att_ema","snap_share_ema","redzone_opps_ema",
              "air_yards_ema","adot_ema","yprr_ema","yards_after_contact","missed_tackles_forced",
              "pressure_rate","opp_dvp_pos_allowed"]:
        df[c] = np.random.uniform(0, 2, n)

    out = validate_and_prepare_features(df, exp, allow_extra=False)
    assert list(out.columns) == exp
    print("✓ test_ok_frame_passes")

def test_binary_violation_fails():
    exp = _schema()
    df = pd.DataFrame({c: [0] for c in exp})
    df["is_favorite"] = 2  # invalid
    try:
        validate_and_prepare_features(df, exp, allow_extra=False)
        assert False, "Expected failure on non-binary is_favorite"
    except ValueError as e:
        assert "binary" in str(e)
        print("✓ test_binary_violation_fails")

def test_injury_exclusivity_fails():
    exp = _schema()
    df = pd.DataFrame({c: [0] for c in exp})
    df["injury_status_Out"] = 1
    df["injury_status_Questionable"] = 1  # duplicate one-hots
    try:
        validate_and_prepare_features(df, exp, allow_extra=False)
        assert False, "Expected failure on injury exclusivity"
    except ValueError as e:
        assert "mutually exclusive" in str(e)
        print("✓ test_injury_exclusivity_fails")

def test_odds_features_present():
    """Test that odds-derived features are present and correctly computed."""
    exp = _schema()
    df = pd.DataFrame({c: [0.0] for c in exp})

    # Set odds features
    df["team_spread"] = -3.5  # Team is favored
    df["team_spread_abs"] = 3.5
    df["total_line"] = 47.0
    df["team_itt"] = 25.25  # (47/2) - (-3.5/2)
    df["is_favorite"] = 1
    df["salary"] = 6000  # Add valid salary

    out = validate_and_prepare_features(df, exp, allow_extra=False)

    assert "team_itt" in out.columns
    assert "team_itt_z" in out.columns
    assert "is_favorite" in out.columns
    assert out["is_favorite"].iloc[0] == 1
    print("✓ test_odds_features_present")

def test_weather_flags_work():
    """Test weather threshold flags."""
    exp = _schema()
    df = pd.DataFrame({c: [0.0] for c in exp})

    df["temperature_f"] = 35  # Cold
    df["wind_mph"] = 20  # High wind
    df["cold_lt40"] = 1
    df["wind_gt15"] = 1
    df["salary"] = 5500  # Add valid salary
    df["total_line"] = 45  # Add valid total
    df["team_spread"] = -3.5
    df["team_spread_abs"] = 3.5
    df["team_itt"] = 25

    out = validate_and_prepare_features(df, exp, allow_extra=False)

    assert out["cold_lt40"].iloc[0] == 1
    assert out["wind_gt15"].iloc[0] == 1
    print("✓ test_weather_flags_work")

def test_feature_order_matches():
    """Test that feature order matches expected schema."""
    exp = _schema()
    df = pd.DataFrame({c: [0.0] for c in reversed(exp)})  # Reverse order
    df["salary"] = 5000  # Add valid salary
    df["total_line"] = 47
    df["team_spread"] = -2
    df["team_spread_abs"] = 2
    df["team_itt"] = 24.5
    df["temperature_f"] = 70

    out = validate_and_prepare_features(df, exp, allow_extra=False)

    assert list(out.columns) == exp
    print("✓ test_feature_order_matches")

if __name__ == "__main__":
    test_ok_frame_passes()
    test_binary_violation_fails()
    test_injury_exclusivity_fails()
    test_odds_features_present()
    test_weather_flags_work()
    test_feature_order_matches()
    print("\n✅ All feature validation tests passed!")
