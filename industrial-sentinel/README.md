# IndustrialSentinel

End-to-end ML pipeline for multivariate industrial sensor streams.
Detects operating regimes, scores anomalies, and predicts near-term sensor values —
all in a single ONNX graph designed for deployment behind a .NET inference service.

---

## Raw data → ONNX in 7 steps

```
1. Simulate / ingest     pump_sim.py generates 6-sensor SCADA-style streams with 4 fault regimes
2. Resample & impute     1 Hz resampling, forward-fill imputation
3. Engineer features     34 features: rolling mean/std/slope, FFT peak, cross-sensor ratios
4. Temporal split        70 / 15 / 15  — strict time order, no shuffling across splits
5. Train                 Multi-task 1D-CNN (PyTorch) + LightGBM baseline for comparison
6. Calibrate thresholds  Conformal prediction per regime → coverage-guaranteed cutoffs
7. Export                ONNX opset 17, parity-validated, inference_contract.json written
```

---

## Problem setup

| Task | Output head | Metric |
|---|---|---|
| Regime classification | softmax(4) | macro F1 |
| Anomaly detection | sigmoid(1) | precision @ 95% recall |
| Short-horizon forecast | linear(10 × 6) | RMSE per sensor |

**Operating regimes**: `HEALTHY`, `CAVITATION`, `BEARING_WEAR`, `SEAL_LEAK`

**Sensors simulated**:

| Sensor | Unit | Key fault indicator |
|---|---|---|
| discharge_pressure | bar | Cavitation, seal leak |
| suction_pressure | bar | Cavitation |
| flow_rate | m³/h | Seal leak, blockage |
| motor_current | A | Bearing wear, overload |
| vibration_rms | mm/s | Bearing wear, imbalance |
| bearing_temp | °C | Bearing wear, lubrication loss |

---

## Model architecture

```
Input  (B, 60, 34)  ← 60-step window of 34 engineered features
   │
   ▼
Conv1D(5, d=1) → Conv1D(5, d=2) → Conv1D(3, d=1) → Conv1D(3, d=2)
   │
GlobalAvgPool → Repr (B, 128)
   │
   ├── Head A  Linear(128→64→4)  + softmax   →  regime_probs  (B, 4)
   ├── Head B  Linear(128→32→1)  + sigmoid   →  anomaly_score (B, 1)
   └── Head C  Linear(128→128→60) reshape    →  forecast      (B, 10, 6)
```

Single ONNX graph — one forward pass returns all three outputs.

---

## Novelty

**1. Regime-conditioned anomaly thresholds**
Anomaly cutoffs adapt per detected operating regime using split conformal prediction,
giving coverage-guaranteed thresholds rather than fixed global cutoffs.

**2. Multi-task single-graph ONNX**
One inference call, three predictions, no pipeline overhead in production.

**3. Machine-readable inference contract**
`artifacts/inference_contract.json` specifies input/output tensor shapes, dtypes,
label order, scaler parameters, and recommended thresholds — ready to paste into
a C# ONNX Runtime integration.

---

## Repo structure

```
industrial-sentinel/
├── sentinel/
│   ├── datagen/      pump_sim.py          — physics-based SCADA generator
│   ├── features/     pipeline.py          — feature engineering + scaler
│   ├── models/       cnn.py, lgbm_baseline.py
│   ├── calibrate/    conformal.py         — per-regime threshold calibration
│   └── export/       onnx_export.py       — ONNX export + contract writer
├── configs/
│   └── train.yaml
├── artifacts/                             — generated after training
│   ├── sentinel.onnx
│   ├── scaler_config.json
│   ├── feature_spec.json
│   ├── inference_contract.json
│   └── thresholds.json
├── data/sim/                              — generated CSVs (.gitignored)
├── train.py
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/you/industrial-sentinel
cd industrial-sentinel
pip install -r requirements.txt

# Train (generates data, engineers features, trains CNN + LGBM, saves artifacts)
python train.py

# Export to ONNX
python -m sentinel.export.onnx_export

# Calibrate anomaly thresholds
python -m sentinel.calibrate.conformal
```

---

## Feature engineering spec (for C# reimplementation)

All 34 features are documented in `artifacts/feature_spec.json` with window sizes,
transform type, and units. The preprocessing sequence for inference:

1. Resample incoming stream to 1 Hz (forward-fill gaps)
2. For each of 6 raw sensors compute: raw value, 30-step rolling mean, 30-step rolling std, 20-step OLS slope, 1-step delta
3. Compute cross-sensor ratios: `dp/sp`, `flow/current`
4. Compute 64-point FFT on vibration window → peak frequency + amplitude
5. Apply StandardScaler: `x_scaled = (x − mean) / std`  (parameters in `scaler_config.json`)
6. Buffer 60 scaled timesteps → pass as `float32[1, 60, 34]` to ONNX Runtime

---

## .NET integration (C# snippet)

```csharp
var session = new InferenceSession("artifacts/sentinel.onnx");

// inputData: float[] of length 1 × 60 × 34, row-major
var tensor = new DenseTensor<float>(inputData, new[] { 1, 60, 34 });
var inputs = new List<NamedOnnxValue> {
    NamedOnnxValue.CreateFromTensor("sensor_window", tensor)
};

using var results = session.Run(inputs);
var regimeLogits  = results.First(r => r.Name == "regime_logits").AsTensor<float>();
var anomalyScore  = results.First(r => r.Name == "anomaly_score").AsTensor<float>();
var forecast      = results.First(r => r.Name == "forecast").AsTensor<float>();
```

Full tensor spec: `artifacts/inference_contract.json`

---

## Why LightGBM alongside the CNN?

The LightGBM baseline uses the same 34 features on individual timesteps (no window).
It typically matches or beats the CNN on regimes that manifest in slow, monotonic
sensor trends (bearing wear temperature rise), but is outperformed on regimes
requiring temporal pattern recognition (cavitation pressure oscillations).
SHAP values from the baseline identify the most predictive features across regimes,
which informed the feature engineering choices.

---

## License

MIT
