"""
Integration tests for SentinelCNN model.

Tests validate:
- Forward pass output shapes: regime_logits (B, 4), anomaly_score (B, 1), forecast (B, 10, 6)
- Forward pass value ranges: anomaly_score in [0, 1], forecast is finite
- Loss computation returns scalar tensor
- Model can overfit small batch (sanity check for gradient flow)
- Model parameter count matches expected architecture

**Validates: Requirements 2.14**
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from sentinel.models.cnn import SentinelCNN, SentinelLoss, N_REGIMES, FORECAST_STEPS, N_RAW_SENSORS


class TestModelForwardPass:
    """Test model forward pass output shapes and value ranges."""

    def test_forward_pass_output_shapes(self):
        """Forward pass should return correct shapes for all three heads."""
        model = SentinelCNN(n_features=34, seq_len=60)
        model.eval()
        
        batch_size = 8
        x = torch.randn(batch_size, 60, 34)
        
        regime_logits, anomaly_score, forecast = model(x)
        
        # Check shapes
        assert regime_logits.shape == (batch_size, N_REGIMES), \
            f"regime_logits should be (B, {N_REGIMES}), got {regime_logits.shape}"
        assert anomaly_score.shape == (batch_size, 1), \
            f"anomaly_score should be (B, 1), got {anomaly_score.shape}"
        assert forecast.shape == (batch_size, FORECAST_STEPS, N_RAW_SENSORS), \
            f"forecast should be (B, {FORECAST_STEPS}, {N_RAW_SENSORS}), got {forecast.shape}"

    def test_forward_pass_batch_size_1(self):
        """Forward pass should work with batch_size=1."""
        model = SentinelCNN(n_features=34, seq_len=60)
        model.eval()
        
        x = torch.randn(1, 60, 34)
        regime_logits, anomaly_score, forecast = model(x)
        
        assert regime_logits.shape == (1, N_REGIMES)
        assert anomaly_score.shape == (1, 1)
        assert forecast.shape == (1, FORECAST_STEPS, N_RAW_SENSORS)

    def test_forward_pass_different_batch_sizes(self):
        """Forward pass should work with various batch sizes."""
        model = SentinelCNN(n_features=34, seq_len=60)
        model.eval()
        
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 60, 34)
            regime_logits, anomaly_score, forecast = model(x)
            
            assert regime_logits.shape[0] == batch_size
            assert anomaly_score.shape[0] == batch_size
            assert forecast.shape[0] == batch_size


class TestModelValueRanges:
    """Test model output value ranges."""

    def test_anomaly_score_in_range_0_1(self):
        """Anomaly score should be in [0, 1] due to sigmoid activation."""
        model = SentinelCNN(n_features=34, seq_len=60)
        model.eval()
        
        # Test with random inputs
        for _ in range(10):
            x = torch.randn(8, 60, 34)
            _, anomaly_score, _ = model(x)
            
            assert (anomaly_score >= 0.0).all(), \
                "anomaly_score should be >= 0.0"
            assert (anomaly_score <= 1.0).all(), \
                "anomaly_score should be <= 1.0"

    def test_forecast_is_finite(self):
        """Forecast values should be finite (no NaN or inf)."""
        model = SentinelCNN(n_features=34, seq_len=60)
        model.eval()
        
        # Test with random inputs
        for _ in range(10):
            x = torch.randn(8, 60, 34)
            _, _, forecast = model(x)
            
            assert torch.isfinite(forecast).all(), \
                "forecast should contain only finite values"

    def test_regime_logits_are_finite(self):
        """Regime logits should be finite (no NaN or inf)."""
        model = SentinelCNN(n_features=34, seq_len=60)
        model.eval()
        
        # Test with random inputs
        for _ in range(10):
            x = torch.randn(8, 60, 34)
            regime_logits, _, _ = model(x)
            
            assert torch.isfinite(regime_logits).all(), \
                "regime_logits should contain only finite values"


class TestLossComputation:
    """Test loss computation."""

    def test_loss_returns_scalar_tensor(self):
        """Loss should return a scalar tensor."""
        model = SentinelCNN(n_features=34, seq_len=60)
        criterion = SentinelLoss()
        
        batch_size = 8
        x = torch.randn(batch_size, 60, 34)
        
        # Create dummy targets
        regime_target = torch.randint(0, N_REGIMES, (batch_size,))
        anomaly_target = torch.rand(batch_size)
        forecast_target = torch.randn(batch_size, FORECAST_STEPS, N_RAW_SENSORS)
        
        # Forward pass
        regime_logits, anomaly_score, forecast = model(x)
        
        # Compute loss
        loss, loss_dict = criterion(
            regime_logits, anomaly_score, forecast,
            regime_target, anomaly_target, forecast_target
        )
        
        # Check loss is scalar
        assert loss.dim() == 0, \
            f"loss should be scalar (0-dim), got shape {loss.shape}"
        assert torch.isfinite(loss), \
            "loss should be finite"

    def test_loss_components_are_finite(self):
        """All loss components should be finite."""
        model = SentinelCNN(n_features=34, seq_len=60)
        criterion = SentinelLoss()
        
        batch_size = 8
        x = torch.randn(batch_size, 60, 34)
        
        # Create dummy targets
        regime_target = torch.randint(0, N_REGIMES, (batch_size,))
        anomaly_target = torch.rand(batch_size)
        forecast_target = torch.randn(batch_size, FORECAST_STEPS, N_RAW_SENSORS)
        
        # Forward pass
        regime_logits, anomaly_score, forecast = model(x)
        
        # Compute loss
        loss, loss_dict = criterion(
            regime_logits, anomaly_score, forecast,
            regime_target, anomaly_target, forecast_target
        )
        
        # Check all components are finite
        assert np.isfinite(loss_dict["loss_regime"]), \
            "loss_regime should be finite"
        assert np.isfinite(loss_dict["loss_anomaly"]), \
            "loss_anomaly should be finite"
        assert np.isfinite(loss_dict["loss_forecast"]), \
            "loss_forecast should be finite"

    def test_loss_is_positive(self):
        """Loss should be positive."""
        model = SentinelCNN(n_features=34, seq_len=60)
        criterion = SentinelLoss()
        
        batch_size = 8
        x = torch.randn(batch_size, 60, 34)
        
        # Create dummy targets
        regime_target = torch.randint(0, N_REGIMES, (batch_size,))
        anomaly_target = torch.rand(batch_size)
        forecast_target = torch.randn(batch_size, FORECAST_STEPS, N_RAW_SENSORS)
        
        # Forward pass
        regime_logits, anomaly_score, forecast = model(x)
        
        # Compute loss
        loss, _ = criterion(
            regime_logits, anomaly_score, forecast,
            regime_target, anomaly_target, forecast_target
        )
        
        assert loss >= 0.0, \
            "loss should be non-negative"


class TestModelOverfitting:
    """Test model can overfit small batch (sanity check for gradient flow)."""

    def test_model_can_overfit_small_batch(self):
        """Model should be able to overfit a small batch (gradient flow sanity check)."""
        model = SentinelCNN(n_features=34, seq_len=60)
        criterion = SentinelLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create small batch
        batch_size = 4
        x = torch.randn(batch_size, 60, 34)
        regime_target = torch.randint(0, N_REGIMES, (batch_size,))
        anomaly_target = torch.rand(batch_size)
        forecast_target = torch.randn(batch_size, FORECAST_STEPS, N_RAW_SENSORS)
        
        # Train for a few iterations
        model.train()
        initial_loss = None
        final_loss = None
        
        for i in range(100):
            optimizer.zero_grad()
            regime_logits, anomaly_score, forecast = model(x)
            loss, _ = criterion(
                regime_logits, anomaly_score, forecast,
                regime_target, anomaly_target, forecast_target
            )
            
            if i == 0:
                initial_loss = loss.item()
            if i == 99:
                final_loss = loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, \
            f"Model should overfit small batch (initial={initial_loss:.4f}, final={final_loss:.4f})"

    def test_gradients_flow_through_all_heads(self):
        """Gradients should flow through all three heads."""
        model = SentinelCNN(n_features=34, seq_len=60)
        criterion = SentinelLoss()
        
        batch_size = 4
        x = torch.randn(batch_size, 60, 34)
        regime_target = torch.randint(0, N_REGIMES, (batch_size,))
        anomaly_target = torch.rand(batch_size)
        forecast_target = torch.randn(batch_size, FORECAST_STEPS, N_RAW_SENSORS)
        
        # Forward pass
        regime_logits, anomaly_score, forecast = model(x)
        loss, _ = criterion(
            regime_logits, anomaly_score, forecast,
            regime_target, anomaly_target, forecast_target
        )
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, \
                    f"Gradient should exist for {name}"
                assert torch.isfinite(param.grad).all(), \
                    f"Gradient should be finite for {name}"


class TestModelParameterCount:
    """Test model parameter count matches expected architecture."""

    def test_parameter_count_is_reasonable(self):
        """Model should have a reasonable number of parameters."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model should have between 100k and 1M parameters (reasonable for this architecture)
        assert 100_000 <= total_params <= 1_000_000, \
            f"Model should have 100k-1M parameters, got {total_params:,}"
        
        # All parameters should be trainable by default
        assert total_params == trainable_params, \
            "All parameters should be trainable by default"

    def test_parameter_count_consistency(self):
        """Parameter count should be consistent across multiple instantiations."""
        model1 = SentinelCNN(n_features=34, seq_len=60)
        model2 = SentinelCNN(n_features=34, seq_len=60)
        
        params1 = sum(p.numel() for p in model1.parameters())
        params2 = sum(p.numel() for p in model2.parameters())
        
        assert params1 == params2, \
            "Parameter count should be consistent across instantiations"

    def test_backbone_has_most_parameters(self):
        """Backbone should have most of the parameters."""
        model = SentinelCNN(n_features=34, seq_len=60)
        
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        
        # Backbone should have at least 50% of parameters
        assert backbone_params >= total_params * 0.5, \
            "Backbone should have at least 50% of parameters"


class TestModelPredict:
    """Test model predict convenience method."""

    def test_predict_returns_dict(self):
        """Predict method should return a dictionary with named predictions."""
        model = SentinelCNN(n_features=34, seq_len=60)
        x = torch.randn(8, 60, 34)
        
        predictions = model.predict(x)
        
        assert isinstance(predictions, dict), \
            "predict should return a dictionary"
        assert "regime_probs" in predictions
        assert "regime_class" in predictions
        assert "anomaly_score" in predictions
        assert "forecast" in predictions

    def test_predict_regime_probs_sum_to_1(self):
        """Regime probabilities should sum to 1 (softmax output)."""
        model = SentinelCNN(n_features=34, seq_len=60)
        x = torch.randn(8, 60, 34)
        
        predictions = model.predict(x)
        regime_probs = predictions["regime_probs"]
        
        # Check probabilities sum to 1
        prob_sums = regime_probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
            "Regime probabilities should sum to 1"

    def test_predict_regime_class_in_range(self):
        """Regime class should be in [0, N_REGIMES-1]."""
        model = SentinelCNN(n_features=34, seq_len=60)
        x = torch.randn(8, 60, 34)
        
        predictions = model.predict(x)
        regime_class = predictions["regime_class"]
        
        assert (regime_class >= 0).all(), \
            "regime_class should be >= 0"
        assert (regime_class < N_REGIMES).all(), \
            f"regime_class should be < {N_REGIMES}"


class TestModelDifferentConfigurations:
    """Test model with different configurations."""

    def test_model_with_different_seq_len(self):
        """Model should work with different sequence lengths."""
        for seq_len in [30, 60, 120]:
            model = SentinelCNN(n_features=34, seq_len=seq_len)
            x = torch.randn(4, seq_len, 34)
            
            regime_logits, anomaly_score, forecast = model(x)
            
            assert regime_logits.shape == (4, N_REGIMES)
            assert anomaly_score.shape == (4, 1)
            assert forecast.shape == (4, FORECAST_STEPS, N_RAW_SENSORS)

    def test_model_with_different_n_features(self):
        """Model should work with different number of features."""
        for n_features in [20, 34, 50]:
            model = SentinelCNN(n_features=n_features, seq_len=60)
            x = torch.randn(4, 60, n_features)
            
            regime_logits, anomaly_score, forecast = model(x)
            
            assert regime_logits.shape == (4, N_REGIMES)
            assert anomaly_score.shape == (4, 1)
            assert forecast.shape == (4, FORECAST_STEPS, N_RAW_SENSORS)

    def test_model_with_different_dropout(self):
        """Model should work with different dropout rates."""
        for dropout in [0.0, 0.3, 0.5]:
            model = SentinelCNN(n_features=34, seq_len=60, dropout=dropout)
            x = torch.randn(4, 60, 34)
            
            regime_logits, anomaly_score, forecast = model(x)
            
            assert regime_logits.shape == (4, N_REGIMES)
            assert anomaly_score.shape == (4, 1)
            assert forecast.shape == (4, FORECAST_STEPS, N_RAW_SENSORS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
