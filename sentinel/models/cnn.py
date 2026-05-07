"""
Multi-task 1D-CNN for industrial sensor streams.

Architecture
------------
Shared 1D convolutional backbone → three output heads:

  Head A — Regime classifier   : softmax(4)     [HEALTHY, CAVITATION, BEARING_WEAR, SEAL_LEAK]
  Head B — Anomaly score       : sigmoid(1)     [0.0 – 1.0]  higher = more anomalous
  Head C — Short-horizon forecast : linear(FORECAST_STEPS × N_SENSORS)  [raw sensor values]

Input shape  : (batch, seq_len, n_features)   e.g. (32, 60, 34)
ONNX outputs : regime_probs, anomaly_score, forecast
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

REGIME_NAMES   = ["HEALTHY", "CAVITATION", "BEARING_WEAR", "SEAL_LEAK"]
N_REGIMES      = len(REGIME_NAMES)
N_RAW_SENSORS  = 6
FORECAST_STEPS = 10


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class SentinelCNN(nn.Module):
    """
    Multi-task 1D-CNN.

    Parameters
    ----------
    n_features      : number of engineered input features (default 34)
    seq_len         : input sequence length (default 60)
    n_regimes       : number of regime classes (default 4)
    forecast_steps  : steps ahead to forecast (default 10)
    n_raw_sensors   : number of raw sensors to forecast (default 6)
    dropout         : dropout probability
    """

    def __init__(
        self,
        n_features: int = 34,
        seq_len: int = 60,
        n_regimes: int = N_REGIMES,
        forecast_steps: int = FORECAST_STEPS,
        n_raw_sensors: int = N_RAW_SENSORS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features     = n_features
        self.seq_len        = seq_len
        self.n_regimes      = n_regimes
        self.forecast_steps = forecast_steps
        self.n_raw_sensors  = n_raw_sensors

        # --- Shared backbone ---
        # Input: (B, n_features, seq_len)  [channels-first for Conv1d]
        self.backbone = nn.Sequential(
            ConvBlock(n_features, 64,  kernel=5, dilation=1),
            ConvBlock(64,          64,  kernel=5, dilation=2),
            ConvBlock(64,          128, kernel=3, dilation=1),
            ConvBlock(128,         128, kernel=3, dilation=2),
            nn.AdaptiveAvgPool1d(1),   # (B, 128, 1)
        )
        self.dropout = nn.Dropout(dropout)
        repr_dim = 128

        # --- Head A: Regime classifier ---
        self.regime_head = nn.Sequential(
            nn.Linear(repr_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_regimes),
        )

        # --- Head B: Anomaly score ---
        self.anomaly_head = nn.Sequential(
            nn.Linear(repr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # --- Head C: Short-horizon forecast ---
        self.forecast_head = nn.Sequential(
            nn.Linear(repr_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, forecast_steps * n_raw_sensors),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (batch, seq_len, n_features)

        Returns
        -------
        regime_logits : (batch, n_regimes)
        anomaly_score : (batch, 1)
        forecast      : (batch, forecast_steps, n_raw_sensors)
        """
        # Conv1d expects (B, C, L)
        x = x.permute(0, 2, 1)
        z = self.backbone(x).squeeze(-1)   # (B, 128)
        z = self.dropout(z)

        regime_logits = self.regime_head(z)
        anomaly_score = self.anomaly_head(z)
        forecast      = self.forecast_head(z).view(
            -1, self.forecast_steps, self.n_raw_sensors
        )
        return regime_logits, anomaly_score, forecast

    def predict(self, x: torch.Tensor) -> dict:
        """Convenience method returning named predictions."""
        self.eval()
        with torch.no_grad():
            logits, anom, fcast = self(x)
        return {
            "regime_probs":  F.softmax(logits, dim=-1),
            "regime_class":  logits.argmax(dim=-1),
            "anomaly_score": anom.squeeze(-1),
            "forecast":      fcast,
        }


class SentinelLoss(nn.Module):
    """
    Combined loss for the three heads.

    L = w_regime * CrossEntropy(regime)
      + w_anomaly * BCELoss(anomaly)
      + w_forecast * MSELoss(forecast)
    """

    def __init__(
        self,
        w_regime: float   = 1.0,
        w_anomaly: float  = 0.5,
        w_forecast: float = 0.3,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.w_regime   = w_regime
        self.w_anomaly  = w_anomaly
        self.w_forecast = w_forecast
        self.ce   = nn.CrossEntropyLoss(weight=class_weights)
        self.bce  = nn.BCELoss()
        self.mse  = nn.MSELoss()

    def forward(
        self,
        regime_logits:  torch.Tensor,
        anomaly_score:  torch.Tensor,
        forecast:       torch.Tensor,
        regime_target:  torch.Tensor,
        anomaly_target: torch.Tensor,
        forecast_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        l_regime   = self.ce(regime_logits, regime_target)
        l_anomaly  = self.bce(anomaly_score.squeeze(-1), anomaly_target)
        l_forecast = self.mse(forecast, forecast_target)

        total = (
            self.w_regime   * l_regime
            + self.w_anomaly  * l_anomaly
            + self.w_forecast * l_forecast
        )
        return total, {
            "loss_regime":   l_regime.item(),
            "loss_anomaly":  l_anomaly.item(),
            "loss_forecast": l_forecast.item(),
        }


if __name__ == "__main__":
    # Quick sanity check
    model = SentinelCNN(n_features=34, seq_len=60)
    x = torch.randn(8, 60, 34)
    logits, anom, fcast = model(x)
    print(f"regime_logits : {logits.shape}")     # (8, 4)
    print(f"anomaly_score : {anom.shape}")       # (8, 1)
    print(f"forecast      : {fcast.shape}")      # (8, 10, 6)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params  : {total_params:,}")
