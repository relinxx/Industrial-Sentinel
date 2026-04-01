"""
ONNX export validation tests.

Tests validate:
- ONNX export completes without errors
- ONNX model output matches PyTorch output within tolerance (atol=1e-5)
- ONNX model accepts correct input shape (1, seq_len, n_features)
- inference_contract.json is valid JSON with required fields

**Validates: Requirements 2.15**
"""

import json
import os
import tempfile
import pytest
import numpy as np
import torch
import onnx
import onnxruntime as ort

from sentinel.models.cnn import SentinelCNN, N_REGIMES, FORECAST_STEPS, N_RAW_SENSORS
from sentinel.export.onnx_export import export_to_onnx, validate_onnx, write_inference_contract


class TestONNXExport:
    """Test ONNX export completes without errors."""

    def test_onnx_export_completes(self):
        """ONNX export should complete without errors."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            
            # Should not raise any exceptions
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # File should exist
            assert os.path.exists(onnx_path), \
                "ONNX file should be created"

    def test_onnx_export_creates_valid_model(self):
        """Exported ONNX model should be valid."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)  # Should not raise

    def test_onnx_export_with_different_batch_sizes(self):
        """ONNX export should work with different batch sizes."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        for batch_size in [1, 4, 8]:
            with tempfile.TemporaryDirectory() as tmpdir:
                onnx_path = os.path.join(tmpdir, f"test_model_bs{batch_size}.onnx")
                export_to_onnx(model, onnx_path, seq_len=60, n_features=34, batch_size=batch_size)
                
                assert os.path.exists(onnx_path)


class TestONNXOutputParity:
    """Test ONNX model output matches PyTorch output."""

    def test_onnx_output_matches_pytorch(self):
        """ONNX model output should match PyTorch output within tolerance."""
        model = SentinelCNN(n_features=34, seq_len=60)
        model.eval()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Test parity
            parity_ok = validate_onnx(model, onnx_path, seq_len=60, n_features=34, atol=1e-5)
            assert parity_ok, \
                "ONNX output should match PyTorch output within tolerance"

    def test_onnx_output_parity_single_sample(self):
        """Test ONNX output parity on a single sample."""
        model = SentinelCNN(n_features=34, seq_len=60)
        model.eval()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Create ONNX Runtime session
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            
            # Create test input
            x_np = np.random.randn(1, 60, 34).astype(np.float32)
            x_pt = torch.from_numpy(x_np)
            
            # PyTorch forward pass
            with torch.no_grad():
                pt_logits, pt_anom, pt_fcast = model(x_pt)
                pt_logits_np = pt_logits.numpy()
                pt_anom_np = pt_anom.numpy()
                pt_fcast_np = pt_fcast.numpy()
            
            # ONNX Runtime forward pass
            ort_out = sess.run(None, {"sensor_window": x_np})
            ort_logits, ort_anom, ort_fcast = ort_out
            
            # Check outputs match
            assert np.allclose(pt_logits_np, ort_logits, atol=1e-5), \
                "regime_logits should match between PyTorch and ONNX"
            assert np.allclose(pt_anom_np, ort_anom, atol=1e-5), \
                "anomaly_score should match between PyTorch and ONNX"
            assert np.allclose(pt_fcast_np, ort_fcast, atol=1e-5), \
                "forecast should match between PyTorch and ONNX"

    def test_onnx_output_parity_multiple_samples(self):
        """Test ONNX output parity on multiple samples."""
        model = SentinelCNN(n_features=34, seq_len=60)
        model.eval()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Test with multiple samples
            parity_ok = validate_onnx(model, onnx_path, seq_len=60, n_features=34, n_samples=20, atol=1e-5)
            assert parity_ok, \
                "ONNX output should match PyTorch output on multiple samples"


class TestONNXInputShape:
    """Test ONNX model accepts correct input shape."""

    def test_onnx_accepts_correct_input_shape(self):
        """ONNX model should accept input shape (1, seq_len, n_features)."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Create ONNX Runtime session
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            
            # Test with correct input shape
            x = np.random.randn(1, 60, 34).astype(np.float32)
            outputs = sess.run(None, {"sensor_window": x})
            
            # Should return 3 outputs
            assert len(outputs) == 3, \
                "ONNX model should return 3 outputs"

    def test_onnx_accepts_different_batch_sizes(self):
        """ONNX model should accept different batch sizes (dynamic axis)."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Create ONNX Runtime session
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            
            # Test with different batch sizes
            for batch_size in [1, 4, 8]:
                x = np.random.randn(batch_size, 60, 34).astype(np.float32)
                outputs = sess.run(None, {"sensor_window": x})
                
                # Check output shapes
                assert outputs[0].shape == (batch_size, N_REGIMES), \
                    f"regime_logits should have batch_size={batch_size}"
                assert outputs[1].shape == (batch_size, 1), \
                    f"anomaly_score should have batch_size={batch_size}"
                assert outputs[2].shape == (batch_size, FORECAST_STEPS, N_RAW_SENSORS), \
                    f"forecast should have batch_size={batch_size}"

    def test_onnx_rejects_wrong_input_shape(self):
        """ONNX model should reject wrong input shape."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Create ONNX Runtime session
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            
            # Test with wrong input shape (wrong n_features)
            x_wrong = np.random.randn(1, 60, 20).astype(np.float32)
            
            with pytest.raises(Exception):
                sess.run(None, {"sensor_window": x_wrong})


class TestONNXOutputShapes:
    """Test ONNX model output shapes."""

    def test_onnx_output_shapes_are_correct(self):
        """ONNX model output shapes should match expected shapes."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Create ONNX Runtime session
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            
            # Test input
            batch_size = 4
            x = np.random.randn(batch_size, 60, 34).astype(np.float32)
            outputs = sess.run(None, {"sensor_window": x})
            
            regime_logits, anomaly_score, forecast = outputs
            
            assert regime_logits.shape == (batch_size, N_REGIMES), \
                f"regime_logits should be ({batch_size}, {N_REGIMES})"
            assert anomaly_score.shape == (batch_size, 1), \
                f"anomaly_score should be ({batch_size}, 1)"
            assert forecast.shape == (batch_size, FORECAST_STEPS, N_RAW_SENSORS), \
                f"forecast should be ({batch_size}, {FORECAST_STEPS}, {N_RAW_SENSORS})"


class TestInferenceContract:
    """Test inference_contract.json is valid and contains required fields."""

    def test_inference_contract_is_valid_json(self):
        """inference_contract.json should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            contract_path = os.path.join(tmpdir, "inference_contract.json")
            
            # Create dummy scaler and feature spec files
            scaler_path = os.path.join(tmpdir, "scaler_config.json")
            feature_spec_path = os.path.join(tmpdir, "feature_spec.json")
            
            with open(scaler_path, "w") as f:
                json.dump({"type": "StandardScaler", "mean": [0.0] * 34, "std": [1.0] * 34}, f)
            
            with open(feature_spec_path, "w") as f:
                json.dump({"version": "1.0", "n_features": 34}, f)
            
            # Write inference contract
            write_inference_contract(
                onnx_path=onnx_path,
                scaler_path=scaler_path,
                feature_spec_path=feature_spec_path,
                seq_len=60,
                n_features=34,
                output_path=contract_path
            )
            
            # Load and validate JSON
            assert os.path.exists(contract_path), \
                "inference_contract.json should be created"
            
            with open(contract_path) as f:
                contract = json.load(f)  # Should not raise
            
            assert isinstance(contract, dict), \
                "inference_contract should be a dictionary"

    def test_inference_contract_has_required_fields(self):
        """inference_contract.json should have all required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            contract_path = os.path.join(tmpdir, "inference_contract.json")
            
            # Create dummy scaler and feature spec files
            scaler_path = os.path.join(tmpdir, "scaler_config.json")
            feature_spec_path = os.path.join(tmpdir, "feature_spec.json")
            
            with open(scaler_path, "w") as f:
                json.dump({"type": "StandardScaler", "mean": [0.0] * 34, "std": [1.0] * 34}, f)
            
            with open(feature_spec_path, "w") as f:
                json.dump({"version": "1.0", "n_features": 34}, f)
            
            # Write inference contract
            write_inference_contract(
                onnx_path=onnx_path,
                scaler_path=scaler_path,
                feature_spec_path=feature_spec_path,
                seq_len=60,
                n_features=34,
                output_path=contract_path
            )
            
            # Load contract
            with open(contract_path) as f:
                contract = json.load(f)
            
            # Check required top-level fields
            required_fields = ["version", "model_file", "onnx_opset", "framework", 
                             "input", "outputs", "preprocessing", "runtime"]
            for field in required_fields:
                assert field in contract, \
                    f"inference_contract should have '{field}' field"

    def test_inference_contract_input_section(self):
        """inference_contract.json input section should have correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            contract_path = os.path.join(tmpdir, "inference_contract.json")
            
            # Create dummy scaler and feature spec files
            scaler_path = os.path.join(tmpdir, "scaler_config.json")
            feature_spec_path = os.path.join(tmpdir, "feature_spec.json")
            
            with open(scaler_path, "w") as f:
                json.dump({"type": "StandardScaler", "mean": [0.0] * 34, "std": [1.0] * 34}, f)
            
            with open(feature_spec_path, "w") as f:
                json.dump({"version": "1.0", "n_features": 34}, f)
            
            # Write inference contract
            write_inference_contract(
                onnx_path=onnx_path,
                scaler_path=scaler_path,
                feature_spec_path=feature_spec_path,
                seq_len=60,
                n_features=34,
                output_path=contract_path
            )
            
            # Load contract
            with open(contract_path) as f:
                contract = json.load(f)
            
            # Check input section
            assert "input" in contract
            input_section = contract["input"]
            
            assert input_section["name"] == "sensor_window"
            assert input_section["dtype"] == "float32"
            assert input_section["seq_len"] == 60
            assert input_section["n_features"] == 34

    def test_inference_contract_outputs_section(self):
        """inference_contract.json outputs section should have correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            contract_path = os.path.join(tmpdir, "inference_contract.json")
            
            # Create dummy scaler and feature spec files
            scaler_path = os.path.join(tmpdir, "scaler_config.json")
            feature_spec_path = os.path.join(tmpdir, "feature_spec.json")
            
            with open(scaler_path, "w") as f:
                json.dump({"type": "StandardScaler", "mean": [0.0] * 34, "std": [1.0] * 34}, f)
            
            with open(feature_spec_path, "w") as f:
                json.dump({"version": "1.0", "n_features": 34}, f)
            
            # Write inference contract
            write_inference_contract(
                onnx_path=onnx_path,
                scaler_path=scaler_path,
                feature_spec_path=feature_spec_path,
                seq_len=60,
                n_features=34,
                output_path=contract_path
            )
            
            # Load contract
            with open(contract_path) as f:
                contract = json.load(f)
            
            # Check outputs section
            assert "outputs" in contract
            outputs = contract["outputs"]
            
            # Check all three outputs are present
            assert "regime_logits" in outputs
            assert "anomaly_score" in outputs
            assert "forecast" in outputs
            
            # Check regime_logits structure
            assert outputs["regime_logits"]["name"] == "regime_logits"
            assert outputs["regime_logits"]["dtype"] == "float32"
            assert "labels" in outputs["regime_logits"]
            
            # Check anomaly_score structure
            assert outputs["anomaly_score"]["name"] == "anomaly_score"
            assert outputs["anomaly_score"]["dtype"] == "float32"
            assert outputs["anomaly_score"]["range"] == [0.0, 1.0]
            
            # Check forecast structure
            assert outputs["forecast"]["name"] == "forecast"
            assert outputs["forecast"]["dtype"] == "float32"
            assert outputs["forecast"]["forecast_steps"] == FORECAST_STEPS


class TestONNXModelMetadata:
    """Test ONNX model metadata."""

    def test_onnx_model_has_correct_opset(self):
        """ONNX model should use correct opset version (17 or higher)."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Check opset version (should be 17 or higher, PyTorch may upgrade to 18+)
            opset_version = onnx_model.opset_import[0].version
            assert opset_version >= 17, \
                f"ONNX model should use opset 17 or higher, got {opset_version}"

    def test_onnx_model_has_correct_input_names(self):
        """ONNX model should have correct input names."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Create ONNX Runtime session
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            
            # Check input names
            input_names = [inp.name for inp in sess.get_inputs()]
            assert "sensor_window" in input_names, \
                "ONNX model should have 'sensor_window' input"

    def test_onnx_model_has_correct_output_names(self):
        """ONNX model should have correct output names."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test_model.onnx")
            export_to_onnx(model, onnx_path, seq_len=60, n_features=34)
            
            # Create ONNX Runtime session
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            
            # Check output names
            output_names = [out.name for out in sess.get_outputs()]
            assert "regime_logits" in output_names, \
                "ONNX model should have 'regime_logits' output"
            assert "anomaly_score" in output_names, \
                "ONNX model should have 'anomaly_score' output"
            assert "forecast" in output_names, \
                "ONNX model should have 'forecast' output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
