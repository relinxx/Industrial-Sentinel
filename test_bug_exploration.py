"""
Bug Condition Exploration Tests for ML Pipeline Fixes

**CRITICAL**: These tests MUST FAIL on unfixed code - failure confirms the bugs exist.
**DO NOT attempt to fix the tests or the code when they fail.**
**NOTE**: These tests encode the expected behavior - they will validate the fixes when they pass after implementation.
**GOAL**: Surface counterexamples that demonstrate the bugs exist.

This test suite validates all six bug categories:
1. Rolling Window Data Leakage
2. Incomplete Evaluation Metrics
3. Calibration Not Integrated
4. Missing LightGBM Baseline
5. No Standalone Evaluation Script
6. No Test Suite

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15**
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sentinel.features.pipeline import engineer_features, ROLL_WINDOW, SLOPE_WINDOW, SENSOR_COLS
from sentinel.datagen.pump_sim import PumpConfig, simulate


class TestCategory1_RollingWindowDataLeakage:
    """
    Category 1: Rolling Window Data Leakage
    
    Tests that rolling mean, rolling std, and slope features include the current timestep,
    demonstrating look-ahead bias (data leakage).
    
    **Expected on UNFIXED code**: Tests FAIL (demonstrates leakage exists)
    **Expected on FIXED code**: Tests PASS (leakage is fixed)
    """
    
    def test_rolling_mean_includes_current_timestep(self):
        """
        Test that rolling mean at timestep t includes value at t (demonstrates leakage).
        
        Create synthetic data: [0, 1, 2, 3, ..., 100] with ROLL_WINDOW=30
        At t=50, rolling mean should be mean([21:51]) = 35.5 if leaky (includes t=50)
        At t=50, rolling mean should be mean([20:50]) = 34.5 if correct (excludes t=50)
        
        **Validates: Requirement 1.1**
        """
        # Create synthetic linear data
        n_samples = 101
        synthetic_data = np.arange(n_samples, dtype=np.float32)
        
        # Create DataFrame with all required sensor columns
        df = pd.DataFrame({col: synthetic_data for col in SENSOR_COLS})
        df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
        
        # Engineer features
        feat = engineer_features(df)
        
        # Check rolling mean at t=50 for first sensor
        sensor = SENSOR_COLS[0]
        roll_mean_at_50 = feat[f"{sensor}_roll_mean"].iloc[50]
        
        # If leaky: mean([21:51]) = mean(21, 22, ..., 50) = 35.5
        # If correct: mean([20:50]) = mean(20, 21, ..., 49) = 34.5
        expected_leaky = np.mean(synthetic_data[21:51])  # includes t=50
        expected_correct = np.mean(synthetic_data[20:50])  # excludes t=50
        
        print(f"\nRolling mean at t=50: {roll_mean_at_50}")
        print(f"Expected if leaky (includes t=50): {expected_leaky}")
        print(f"Expected if correct (excludes t=50): {expected_correct}")
        
        # On UNFIXED code, this should be close to expected_leaky (35.5)
        # On FIXED code, this should be close to expected_correct (34.5)
        # We assert the CORRECT behavior, so this FAILS on unfixed code
        assert np.isclose(roll_mean_at_50, expected_correct, atol=0.01), \
            f"Rolling mean includes current timestep (leakage detected): got {roll_mean_at_50}, expected {expected_correct}"
    
    def test_rolling_std_includes_current_timestep(self):
        """
        Test that rolling std at timestep t includes value at t (demonstrates leakage).
        
        **Validates: Requirement 1.2**
        """
        # Create synthetic data with known variance pattern
        n_samples = 101
        synthetic_data = np.arange(n_samples, dtype=np.float32)
        
        df = pd.DataFrame({col: synthetic_data for col in SENSOR_COLS})
        df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
        
        feat = engineer_features(df)
        
        sensor = SENSOR_COLS[0]
        roll_std_at_50 = feat[f"{sensor}_roll_std"].iloc[50]
        
        # If leaky: std([21:51]) includes t=50
        # If correct: std([20:50]) excludes t=50
        expected_leaky = np.std(synthetic_data[21:51], ddof=1)
        expected_correct = np.std(synthetic_data[20:50], ddof=1)
        
        print(f"\nRolling std at t=50: {roll_std_at_50}")
        print(f"Expected if leaky (includes t=50): {expected_leaky}")
        print(f"Expected if correct (excludes t=50): {expected_correct}")
        
        # Assert CORRECT behavior (will FAIL on unfixed code)
        assert np.isclose(roll_std_at_50, expected_correct, atol=0.01), \
            f"Rolling std includes current timestep (leakage detected): got {roll_std_at_50}, expected {expected_correct}"
    
    def test_slope_includes_current_timestep(self):
        """
        Test that slope at timestep t includes value at t (demonstrates leakage).
        
        **Validates: Requirement 1.3**
        """
        # Create synthetic linear trend
        n_samples = 101
        synthetic_data = np.arange(n_samples, dtype=np.float32) * 2.0  # slope = 2.0
        
        df = pd.DataFrame({col: synthetic_data for col in SENSOR_COLS})
        df['timestamp'] = pd.date_range('2024-01-01', periods=n_samples, freq='1s')
        
        feat = engineer_features(df)
        
        sensor = SENSOR_COLS[0]
        slope_at_50 = feat[f"{sensor}_slope"].iloc[50]
        
        # For linear data with slope=2.0, the computed slope should be ~2.0
        # But if leaky, it uses window [31:51] (includes t=50)
        # If correct, it uses window [30:50] (excludes t=50)
        # For linear data, both should give ~2.0, so we need to check the window indices
        
        # Let's use a different approach: check if the slope calculation
        # is using the correct window by examining the implementation
        # For now, we'll check that slope is computed correctly
        
        print(f"\nSlope at t=50: {slope_at_50}")
        print(f"Expected slope for linear data: 2.0")
        
        # The key issue is whether the window includes t or not
        # For a perfect linear trend, slope should be exactly 2.0
        # We'll verify by checking the window used in the calculation
        
        # Create a step function to detect leakage more clearly
        step_data = np.concatenate([np.zeros(50), np.ones(51)])
        df_step = pd.DataFrame({col: step_data for col in SENSOR_COLS})
        df_step['timestamp'] = pd.date_range('2024-01-01', periods=101, freq='1s')
        
        feat_step = engineer_features(df_step)
        slope_at_50_step = feat_step[f"{sensor}_slope"].iloc[50]
        
        # At t=50 (first 1.0 value):
        # If leaky: window [31:51] includes some 1.0 values → positive slope
        # If correct: window [30:50] only has 0.0 values → slope near 0
        
        print(f"Slope at t=50 for step function: {slope_at_50_step}")
        print(f"Expected if leaky (includes t=50): positive slope")
        print(f"Expected if correct (excludes t=50): slope near 0")
        
        # Assert CORRECT behavior (will FAIL on unfixed code if leaky)
        assert slope_at_50_step <= 0.01, \
            f"Slope includes current timestep (leakage detected): got {slope_at_50_step}, expected near 0"


class TestCategory2_IncompleteEvaluationMetrics:
    """
    Category 2: Incomplete Evaluation Metrics
    
    Tests that comprehensive metrics are NOT computed and saved after training.
    
    **Expected on UNFIXED code**: Tests FAIL (metrics file doesn't exist)
    **Expected on FIXED code**: Tests PASS (metrics file exists with all required metrics)
    """
    
    def test_test_metrics_json_not_exists(self):
        """
        Test that artifacts/test_metrics.json does NOT exist after training.
        
        **Validates: Requirements 1.4, 1.5, 1.6, 1.7**
        """
        artifacts_dir = Path("artifacts")
        test_metrics_path = artifacts_dir / "test_metrics.json"
        
        # Check if test_metrics.json exists
        exists = test_metrics_path.exists()
        
        print(f"\ntest_metrics.json exists: {exists}")
        
        if exists:
            # If it exists, check if it has comprehensive metrics
            with open(test_metrics_path) as f:
                metrics = json.load(f)
            
            print(f"Metrics keys: {list(metrics.keys())}")
            
            # Check for required metrics
            required_keys = [
                "macro_f1",           # Regime classification
                "confusion_matrix",   # Regime classification
                "auroc",              # Anomaly detection
                "auprc",              # Anomaly detection
                "precision_at_95_recall",  # Anomaly detection
                "per_sensor_rmse",    # Forecasting
                "per_sensor_mae",     # Forecasting
            ]
            
            missing_keys = [key for key in required_keys if key not in metrics]
            
            assert len(missing_keys) == 0, \
                f"test_metrics.json exists but missing required keys: {missing_keys}"
        else:
            # On unfixed code, file should NOT exist
            # We assert it SHOULD exist (correct behavior), so this FAILS on unfixed code
            pytest.fail("test_metrics.json does not exist (incomplete evaluation metrics)")


class TestCategory3_CalibrationNotIntegrated:
    """
    Category 3: Calibration Not Integrated
    
    Tests that conformal calibration is NOT run after training.
    
    **Expected on UNFIXED code**: Tests FAIL (thresholds.json doesn't exist)
    **Expected on FIXED code**: Tests PASS (thresholds.json exists)
    """
    
    def test_thresholds_json_not_exists(self):
        """
        Test that artifacts/thresholds.json does NOT exist after training.
        
        **Validates: Requirements 1.8, 1.9**
        """
        artifacts_dir = Path("artifacts")
        thresholds_path = artifacts_dir / "thresholds.json"
        
        exists = thresholds_path.exists()
        
        print(f"\nthresholds.json exists: {exists}")
        
        if exists:
            # If it exists, verify it has the correct structure
            with open(thresholds_path) as f:
                thresholds = json.load(f)
            
            print(f"Thresholds keys: {list(thresholds.keys())}")
            
            required_keys = ["threshold", "coverage", "n_healthy"]
            missing_keys = [key for key in required_keys if key not in thresholds]
            
            assert len(missing_keys) == 0, \
                f"thresholds.json exists but missing required keys: {missing_keys}"
        else:
            # On unfixed code, file should NOT exist
            # We assert it SHOULD exist (correct behavior), so this FAILS on unfixed code
            pytest.fail("thresholds.json does not exist (calibration not integrated)")
    
    def test_run_calibration_function_exists_but_not_called(self):
        """
        Test that run_calibration() function exists but is never called in train.py.
        
        **Validates: Requirement 1.8**
        """
        # Check that the function exists
        from sentinel.calibrate.conformal import run_calibration
        assert callable(run_calibration), "run_calibration function should exist"
        
        # Check that train.py doesn't import or call it
        train_py_path = Path("train.py")
        with open(train_py_path) as f:
            train_code = f.read()
        
        has_import = "from sentinel.calibrate.conformal import run_calibration" in train_code
        has_call = "run_calibration(" in train_code
        
        print(f"\ntrain.py imports run_calibration: {has_import}")
        print(f"train.py calls run_calibration: {has_call}")
        
        # On unfixed code, these should be False
        # We assert they SHOULD be True (correct behavior), so this FAILS on unfixed code
        assert has_import, "train.py should import run_calibration"
        assert has_call, "train.py should call run_calibration"


class TestCategory4_MissingLightGBMBaseline:
    """
    Category 4: Missing LightGBM Baseline
    
    Tests that LightGBM baseline is NOT trained during training.
    
    **Expected on UNFIXED code**: Tests FAIL (lgbm files don't exist)
    **Expected on FIXED code**: Tests PASS (lgbm files exist)
    """
    
    def test_lgbm_baseline_not_exists(self):
        """
        Test that artifacts/lgbm_baseline.pkl does NOT exist after training.
        
        **Validates: Requirements 1.10, 1.11**
        """
        artifacts_dir = Path("artifacts")
        lgbm_path = artifacts_dir / "lgbm_baseline.pkl"
        
        exists = lgbm_path.exists()
        
        print(f"\nlgbm_baseline.pkl exists: {exists}")
        
        # On unfixed code, file should NOT exist
        # We assert it SHOULD exist (correct behavior), so this FAILS on unfixed code
        assert exists, "lgbm_baseline.pkl does not exist (LightGBM baseline not trained)"
    
    def test_lgbm_eval_json_not_exists(self):
        """
        Test that artifacts/lgbm_eval.json does NOT exist after training.
        
        **Validates: Requirement 1.11**
        """
        artifacts_dir = Path("artifacts")
        lgbm_eval_path = artifacts_dir / "lgbm_eval.json"
        
        exists = lgbm_eval_path.exists()
        
        print(f"\nlgbm_eval.json exists: {exists}")
        
        assert exists, "lgbm_eval.json does not exist (LightGBM evaluation not run)"
    
    def test_train_lgbm_function_exists_but_not_called(self):
        """
        Test that train_lgbm() function exists but is never called in train.py.
        
        **Validates: Requirement 1.10**
        """
        # Check that the function exists
        from sentinel.models.lgbm_baseline import train_lgbm
        assert callable(train_lgbm), "train_lgbm function should exist"
        
        # Check that train.py doesn't import or call it
        train_py_path = Path("train.py")
        with open(train_py_path) as f:
            train_code = f.read()
        
        has_import = "from sentinel.models.lgbm_baseline import train_lgbm" in train_code
        has_call = "train_lgbm(" in train_code
        
        print(f"\ntrain.py imports train_lgbm: {has_import}")
        print(f"train.py calls train_lgbm: {has_call}")
        
        # On unfixed code, these should be False
        # We assert they SHOULD be True (correct behavior), so this FAILS on unfixed code
        assert has_import, "train.py should import train_lgbm"
        assert has_call, "train.py should call train_lgbm"


class TestCategory5_NoStandaloneEvaluationScript:
    """
    Category 5: No Standalone Evaluation Script
    
    Tests that evaluate.py does NOT exist.
    
    **Expected on UNFIXED code**: Tests FAIL (evaluate.py doesn't exist)
    **Expected on FIXED code**: Tests PASS (evaluate.py exists)
    """
    
    def test_evaluate_script_not_exists(self):
        """
        Test that evaluate.py does NOT exist in repository root.
        
        **Validates: Requirement 1.12**
        """
        evaluate_path = Path("evaluate.py")
        
        exists = evaluate_path.exists()
        
        print(f"\nevaluate.py exists: {exists}")
        
        # On unfixed code, file should NOT exist
        # We assert it SHOULD exist (correct behavior), so this FAILS on unfixed code
        assert exists, "evaluate.py does not exist (no standalone evaluation script)"


class TestCategory6_NoTestSuite:
    """
    Category 6: No Test Suite
    
    Tests that test suite does NOT exist.
    
    **Expected on UNFIXED code**: Tests FAIL (tests/ directory doesn't exist)
    **Expected on FIXED code**: Tests PASS (tests/ directory exists with test files)
    """
    
    def test_tests_directory_not_exists(self):
        """
        Test that tests/ directory does NOT exist.
        
        **Validates: Requirements 1.13, 1.14, 1.15**
        """
        tests_dir = Path("tests")
        
        exists = tests_dir.exists() and tests_dir.is_dir()
        
        print(f"\ntests/ directory exists: {exists}")
        
        # On unfixed code, directory should NOT exist
        # We assert it SHOULD exist (correct behavior), so this FAILS on unfixed code
        assert exists, "tests/ directory does not exist (no test suite)"
    
    def test_test_features_not_exists(self):
        """
        Test that tests/test_features.py does NOT exist.
        
        **Validates: Requirement 1.13**
        """
        test_features_path = Path("tests/test_features.py")
        
        exists = test_features_path.exists()
        
        print(f"\ntests/test_features.py exists: {exists}")
        
        assert exists, "tests/test_features.py does not exist (no feature engineering tests)"
    
    def test_test_model_not_exists(self):
        """
        Test that tests/test_model.py does NOT exist.
        
        **Validates: Requirement 1.14**
        """
        test_model_path = Path("tests/test_model.py")
        
        exists = test_model_path.exists()
        
        print(f"\ntests/test_model.py exists: {exists}")
        
        assert exists, "tests/test_model.py does not exist (no model integration tests)"
    
    def test_test_onnx_not_exists(self):
        """
        Test that tests/test_onnx.py does NOT exist.
        
        **Validates: Requirement 1.15**
        """
        test_onnx_path = Path("tests/test_onnx.py")
        
        exists = test_onnx_path.exists()
        
        print(f"\ntests/test_onnx.py exists: {exists}")
        
        assert exists, "tests/test_onnx.py does not exist (no ONNX validation tests)"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
