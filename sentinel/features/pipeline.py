"""
Feature engineering pipeline for industrial sensor streams.

All transforms are documented in feature_spec.json for C# reimplementation.
No leakage: all rolling features use only past values (closed='left' semantics
are handled by computing on [t-window : t-1]).

Features produced per raw sensor (6 sensors × 5 base features = 30)
---------------------------------------------------------------------
  {sensor}_raw          - z-scored raw value
  {sensor}_roll_mean    - rolling mean over ROLL_WINDOW steps
  {sensor}_roll_std     - rolling std  over ROLL_WINDOW steps
  {sensor}_slope        - linear regression slope over SLOPE_WINDOW steps
  {sensor}_delta        - diff from previous step

Extra cross-sensor / spectral features (4)
-------------------------------------------
  dp_sp_ratio           - discharge / suction pressure ratio
  flow_current_ratio    - flow / motor_current efficiency proxy
  vib_fft_peak_freq     - dominant FFT bin of vibration_rms window
  vib_fft_peak_amp      - amplitude at dominant FFT bin

Total features: 34
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SENSOR_COLS = [
    "discharge_pressure",
    "suction_pressure",
    "flow_rate",
    "motor_current",
    "vibration_rms",
    "bearing_temp",
]

ROLL_WINDOW  = 30   # steps  (~30 s at 1 Hz)
SLOPE_WINDOW = 20   # steps for slope estimation
FFT_WINDOW   = 64   # power-of-2 for FFT


def _slope(arr: np.ndarray) -> float:
    """Linear regression slope of 1-D array."""
    x = np.arange(len(arr), dtype=np.float32)
    if len(arr) < 2:
        return 0.0
    x -= x.mean()
    denom = (x * x).sum()
    return float((x * arr).sum() / denom) if denom != 0 else 0.0


def _fft_peak(arr: np.ndarray, sample_rate_hz: float = 1.0):
    """Return (peak_freq_hz, peak_amplitude) of FFT of 1-D array."""
    n = len(arr)
    if n < 4:
        return 0.0, 0.0
    windowed = arr * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    peak_idx = np.argmax(spectrum[1:]) + 1   # skip DC
    return float(freqs[peak_idx]), float(spectrum[peak_idx])


def engineer_features(df: pd.DataFrame, sample_rate_hz: float = 1.0) -> pd.DataFrame:
    """
    Compute all features from a raw sensor DataFrame.

    Parameters
    ----------
    df : DataFrame with columns in SENSOR_COLS plus 'timestamp'
    sample_rate_hz : sampling rate of the data

    Returns
    -------
    feat : DataFrame of engineered features, same row count as df.
           First ROLL_WINDOW rows will contain NaN — drop before training.
    """
    feat = pd.DataFrame(index=df.index)

    for col in SENSOR_COLS:
        s = df[col].values.astype(np.float32)

        feat[f"{col}_raw"]       = s
        feat[f"{col}_roll_mean"] = (
            pd.Series(s).rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).mean().values
        )
        feat[f"{col}_roll_std"]  = (
            pd.Series(s).rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).std().values
        )
        feat[f"{col}_delta"]     = np.concatenate([[0.0], np.diff(s)])

        # Slope — computed per-row using a trailing window
        slopes = np.full(len(s), np.nan)
        for i in range(SLOPE_WINDOW, len(s)):
            slopes[i] = _slope(s[i - SLOPE_WINDOW: i])
        feat[f"{col}_slope"] = slopes

    # Cross-sensor features
    dp = df["discharge_pressure"].values.astype(np.float32)
    sp = df["suction_pressure"].values.astype(np.float32)
    fl = df["flow_rate"].values.astype(np.float32)
    cu = df["motor_current"].values.astype(np.float32)
    vb = df["vibration_rms"].values.astype(np.float32)

    feat["dp_sp_ratio"]        = np.where(sp > 0, dp / sp, np.nan)
    feat["flow_current_ratio"] = np.where(cu > 0, fl / cu, np.nan)

    # FFT features on vibration — trailing FFT_WINDOW
    fft_peak_freq = np.full(len(vb), np.nan)
    fft_peak_amp  = np.full(len(vb), np.nan)
    for i in range(FFT_WINDOW, len(vb)):
        window = vb[i - FFT_WINDOW: i]
        fft_peak_freq[i], fft_peak_amp[i] = _fft_peak(window, sample_rate_hz)
    feat["vib_fft_peak_freq"] = fft_peak_freq
    feat["vib_fft_peak_amp"]  = fft_peak_amp

    return feat


def build_feature_spec() -> dict:
    """Return a machine-readable feature spec for C# reimplementation."""
    features = []
    for col in SENSOR_COLS:
        features += [
            {"name": f"{col}_raw",       "type": "raw",        "sensor": col, "window": None},
            {"name": f"{col}_roll_mean", "type": "rolling_mean","sensor": col, "window": ROLL_WINDOW},
            {"name": f"{col}_roll_std",  "type": "rolling_std", "sensor": col, "window": ROLL_WINDOW},
            {"name": f"{col}_slope",     "type": "ols_slope",   "sensor": col, "window": SLOPE_WINDOW},
            {"name": f"{col}_delta",     "type": "first_diff",  "sensor": col, "window": 1},
        ]
    features += [
        {"name": "dp_sp_ratio",        "type": "ratio",       "sensors": ["discharge_pressure", "suction_pressure"], "window": None},
        {"name": "flow_current_ratio", "type": "ratio",       "sensors": ["flow_rate", "motor_current"], "window": None},
        {"name": "vib_fft_peak_freq",  "type": "fft_peak_freq", "sensor": "vibration_rms", "window": FFT_WINDOW, "sample_rate_hz": 1.0},
        {"name": "vib_fft_peak_amp",   "type": "fft_peak_amp",  "sensor": "vibration_rms", "window": FFT_WINDOW, "sample_rate_hz": 1.0},
    ]
    return {
        "version": "1.0",
        "n_features": len(features),
        "roll_window": ROLL_WINDOW,
        "slope_window": SLOPE_WINDOW,
        "fft_window": FFT_WINDOW,
        "warmup_steps": max(ROLL_WINDOW, SLOPE_WINDOW, FFT_WINDOW),
        "features": features,
        "notes": (
            "All rolling features use strictly past values (no look-ahead). "
            "Warmup_steps rows at the start of any stream must be discarded "
            "before passing to the model."
        ),
    }


class FeaturePipeline:
    """
    Fit-transform wrapper that applies feature engineering
    and StandardScaler scaling.  Scaler config is exportable to JSON.
    """

    def __init__(self, sample_rate_hz: float = 1.0):
        self.sample_rate_hz = sample_rate_hz
        self.scaler = StandardScaler()
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame, labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        feat = engineer_features(df, self.sample_rate_hz)

        # Drop warmup rows (NaN-heavy)
        warmup = max(ROLL_WINDOW, SLOPE_WINDOW, FFT_WINDOW)
        feat = feat.iloc[warmup:].copy()
        labels = labels.iloc[warmup:].copy()

        # Forward-fill then drop any remaining NaN
        feat = feat.ffill().dropna()
        labels = labels.loc[feat.index]

        self.feature_names_ = list(feat.columns)
        X = self.scaler.fit_transform(feat.values.astype(np.float32))
        self._fitted = True
        return X, labels.values

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self._fitted, "Call fit_transform first"
        feat = engineer_features(df, self.sample_rate_hz)
        feat = feat.ffill().fillna(0.0)
        feat = feat[self.feature_names_]
        return self.scaler.transform(feat.values.astype(np.float32))

    def scaler_config(self) -> dict:
        """Export scaler parameters for C# reimplementation."""
        return {
            "type": "StandardScaler",
            "n_features": len(self.feature_names_),
            "feature_names": self.feature_names_,
            "mean": self.scaler.mean_.tolist(),
            "std": self.scaler.scale_.tolist(),
        }

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FeaturePipeline":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    spec = build_feature_spec()
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/feature_spec.json", "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Feature spec written → {spec['n_features']} features")
