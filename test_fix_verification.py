"""
Quick verification test for Task 3.1: Rolling Window Data Leakage Fix

This test verifies that the fix correctly shifts rolling window calculations
by 1 timestep to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sentinel.features.pipeline import engineer_features, SENSOR_COLS, ROLL_WINDOW, SLOPE_WINDOW


def test_rolling_mean_fix():
    """Verify rolling mean uses only past values (excludes current timestep)."""
    # Create synthetic linear data: [0, 1, 2, 3, ..., 100]
    n_samples = 101
    synthetic_data = np.arange(n_samples, dtype=np.float32)
    
    df = pd.DataFrame({col: synthetic_data for col in SENSOR_COLS})
    df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
    
    feat = engineer_features(df)
    
    # Check rolling mean at t=50 for first sensor
    sensor = SENSOR_COLS[0]
    roll_mean_at_50 = feat[f"{sensor}_roll_mean"].iloc[50]
    
    # After fix: should use shifted values, so at t=50 we compute mean of [20:50] (excludes t=50)
    # The shift(1) moves everything down by 1, so at position 50 we have value from position 49
    # Then rolling window of 30 looks back from position 49, giving us [20:50] range
    expected_correct = np.mean(synthetic_data[20:50])  # mean(20, 21, ..., 49) = 34.5
    
    print(f"\nRolling mean at t=50: {roll_mean_at_50}")
    print(f"Expected (excludes t=50): {expected_correct}")
    
    assert np.isclose(roll_mean_at_50, expected_correct, atol=0.01), \
        f"Rolling mean should exclude current timestep: got {roll_mean_at_50}, expected {expected_correct}"
    print("✓ Rolling mean fix verified!")


def test_rolling_std_fix():
    """Verify rolling std uses only past values (excludes current timestep)."""
    n_samples = 101
    synthetic_data = np.arange(n_samples, dtype=np.float32)
    
    df = pd.DataFrame({col: synthetic_data for col in SENSOR_COLS})
    df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
    
    feat = engineer_features(df)
    
    sensor = SENSOR_COLS[0]
    roll_std_at_50 = feat[f"{sensor}_roll_std"].iloc[50]
    
    # After fix: should use [20:50] (excludes t=50)
    expected_correct = np.std(synthetic_data[20:50], ddof=1)
    
    print(f"\nRolling std at t=50: {roll_std_at_50}")
    print(f"Expected (excludes t=50): {expected_correct}")
    
    assert np.isclose(roll_std_at_50, expected_correct, atol=0.01), \
        f"Rolling std should exclude current timestep: got {roll_std_at_50}, expected {expected_correct}"
    print("✓ Rolling std fix verified!")


def test_slope_fix():
    """Verify slope uses only past values (window ending at t-1)."""
    # Create step function to clearly demonstrate leakage
    n_samples = 101
    step_data = np.concatenate([np.zeros(50), np.ones(51)]).astype(np.float32)
    
    df = pd.DataFrame({col: step_data for col in SENSOR_COLS})
    df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
    
    feat = engineer_features(df)
    
    sensor = SENSOR_COLS[0]
    slope_at_50 = feat[f"{sensor}_slope"].iloc[50]
    
    # At t=50 (first 1.0 value):
    # After fix: window [30:50] uses indices 29 to 49 (all zeros) → slope near 0
    # The fix changes from s[i - SLOPE_WINDOW: i] to s[i - SLOPE_WINDOW - 1: i - 1]
    # At i=50: s[29:50] → all zeros → slope near 0
    
    print(f"\nSlope at t=50 for step function: {slope_at_50}")
    print(f"Expected (excludes t=50): near 0")
    
    assert abs(slope_at_50) <= 0.01, \
        f"Slope should exclude current timestep: got {slope_at_50}, expected near 0"
    print("✓ Slope fix verified!")


if __name__ == "__main__":
    test_rolling_mean_fix()
    test_rolling_std_fix()
    test_slope_fix()
    print("\n✓ All fixes verified successfully!")
