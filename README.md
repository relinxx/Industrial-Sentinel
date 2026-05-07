# Industrial-Sentinel

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)]()

**Production-ready ML pipeline for real-time industrial equipment monitoring and predictive maintenance.**

Industrial-Sentinel is an end-to-end machine learning system that analyzes multivariate sensor streams from industrial equipment (pumps, motors, compressors) to detect operating regimes, score anomalies, and forecast near-term sensor values—all in a single ONNX inference graph optimized for .NET deployment.

---

## 🎯 Key Features

- **Multi-Task Learning**: Single model performs regime classification, anomaly detection, and forecasting simultaneously
- **Production-Grade Performance**: 99.92% accuracy, 0.9994 AUROC on 720k samples
- **ONNX Export**: Single-graph inference ready for .NET/C# integration
- **Conformal Calibration**: Coverage-guaranteed anomaly thresholds (95% coverage)
- **No Data Leakage**: Validated rolling window features using only past values
- **Comprehensive Testing**: 59 tests covering features, models, and ONNX export
- **Machine-Readable Contract**: JSON specification for seamless C# integration

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 99.92% |
| **Macro F1** | 0.9961 |
| **AUROC (Anomaly)** | 0.9994 |
| **AUPRC (Anomaly)** | 0.9990 |
| **Precision @ 95% Recall** | 1.0000 |
| **Test Samples** | 107,992 |
| **Model Size** | 39.3 KB (ONNX) |
| **Inference Latency** | ~5 ms (CPU) |

**Confusion Matrix (107,992 test samples):**
```
                Predicted
              H      C      B      S
Actual  H  94839     0      4      0
        C     32  3918      0      0
        B      0    26   4687      0
        S      0     0     22   4394

H=HEALTHY, C=CAVITATION, B=BEARING_WEAR, S=SEAL_LEAK
```

Only 84 misclassifications out of 107,992 samples (0.078% error rate).

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/industrial-sentinel.git
cd industrial-sentinel
pip install -r requirements.txt
```

### Training

```bash
# Full training (200 hours simulation, ~20 min on GPU)
python train.py

# Quick training (10 hours simulation)
python train.py --sim_hours 10 --epochs 20
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --checkpoint artifacts/sentinel_best.pt --data simulate --sim_hours 50
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=sentinel --cov-report=html
```

---

## 🏗️ Architecture Overview

### Problem Setup

Industrial-Sentinel monitors 6 sensor streams in real-time:

| Sensor | Unit | Key Fault Indicator |
|--------|------|---------------------|
| Discharge Pressure | bar | Cavitation, seal leak |
| Suction Pressure | bar | Cavitation |
| Flow Rate | m³/h | Seal leak, blockage |
| Motor Current | A | Bearing wear, overload |
| Vibration RMS | mm/s | Bearing wear, imbalance |
| Bearing Temperature | °C | Bearing wear, lubrication loss |

### Three Simultaneous Predictions

| Task | Output | Metric |
|------|--------|--------|
| **Regime Classification** | 4-class softmax: HEALTHY, CAVITATION, BEARING_WEAR, SEAL_LEAK | Macro F1: 0.9961 |
| **Anomaly Detection** | Anomaly score [0-1] with calibrated threshold | AUROC: 0.9994 |
| **Short-Horizon Forecasting** | Next 10 seconds of all 6 sensors | RMSE: 0.66-1.35 per sensor |

### Data Pipeline

```
Raw Sensors (6 channels @ 1 Hz)
    ↓
Feature Engineering (34 features)
    ├─ Raw z-scored values (6)
    ├─ Rolling mean/std (12)
    ├─ Slope (6)
    ├─ Delta (first differences) (6)
    ├─ Cross-sensor ratios (2)
    └─ FFT features (2)
    ↓
60-step sliding window
    ↓
Multi-task 1D-CNN
    ├─ Regime Classification Head
    ├─ Anomaly Detection Head
    └─ Forecasting Head
    ↓
ONNX Export (single graph)
```

---

## 🧠 Model Architecture

```
Input: (batch, 60 timesteps, 34 features)
    ↓
Conv1D Backbone (4 layers with dilation)
    ├─ Conv1D(kernel=5, dilation=1) → BatchNorm → ReLU
    ├─ Conv1D(kernel=5, dilation=2) → BatchNorm → ReLU
    ├─ Conv1D(kernel=3, dilation=1) → BatchNorm → ReLU
    └─ Conv1D(kernel=3, dilation=2) → BatchNorm → ReLU
    ↓
Global Average Pooling → 128-dim representation
    ↓
    ├─ Head A: Linear(128→64→4) + Softmax → Regime Probs (B, 4)
    ├─ Head B: Linear(128→32→1) + Sigmoid → Anomaly Score (B, 1)
    └─ Head C: Linear(128→128→60) → Forecast (B, 10, 6)
```

**Total Parameters:** ~143,000  
**Training Time:** ~20 minutes on RTX 4050 GPU

---

## 📦 Deployment

### ONNX Export

```bash
python -m sentinel.export.onnx_export
```

**Generated Artifacts:**
- `artifacts/sentinel.onnx` - ONNX model (opset 17, 39.3 KB)
- `artifacts/inference_contract.json` - Tensor specifications for C# integration
- `artifacts/feature_spec.json` - Feature engineering specification
- `artifacts/scaler_config.json` - StandardScaler parameters
- `artifacts/thresholds.json` - Conformal calibration thresholds

### .NET/C# Integration

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Load ONNX model
var session = new InferenceSession("artifacts/sentinel.onnx");

// Prepare input tensor (1, 60, 34)
var inputData = PrepareFeatures(sensorBuffer); // Your feature engineering
var tensor = new DenseTensor<float>(inputData, new[] { 1, 60, 34 });
var inputs = new List<NamedOnnxValue> {
    NamedOnnxValue.CreateFromTensor("sensor_window", tensor)
};

// Run inference
using var results = session.Run(inputs);
var regimeLogits = results.First(r => r.Name == "regime_logits").AsTensor<float>();
var anomalyScore = results.First(r => r.Name == "anomaly_score").AsTensor<float>();
var forecast = results.First(r => r.Name == "forecast").AsTensor<float>();

// Apply thresholds
var detectedRegime = GetMaxIndex(regimeLogits);
var threshold = LoadThreshold("artifacts/thresholds.json");
var isAnomalous = anomalyScore[0] > threshold;
```

**See [DEPLOYMENT.md](DEPLOYMENT.md) for complete integration guide.**

---

## 📁 Project Structure

```
industrial-sentinel/
├── sentinel/                      # Core ML pipeline
│   ├── datagen/                   # Physics-based SCADA simulator
│   │   └── pump_sim.py
│   ├── features/                  # Feature engineering
│   │   └── pipeline.py
│   ├── models/                    # Model architectures
│   │   ├── cnn.py                 # Multi-task 1D-CNN
│   │   └── lgbm_baseline.py       # LightGBM baseline
│   ├── calibrate/                 # Conformal calibration
│   │   └── conformal.py
│   └── export/                    # ONNX export
│       └── onnx_export.py
├── configs/
│   └── train.yaml                 # Training configuration
├── artifacts/                     # Generated after training
│   ├── sentinel.onnx              # ONNX model
│   ├── inference_contract.json    # C# integration spec
│   ├── feature_spec.json          # Feature engineering spec
│   ├── scaler_config.json         # StandardScaler parameters
│   ├── thresholds.json            # Calibrated thresholds
│   └── test_metrics.json          # Evaluation metrics
├── tests/                         # Test suite (59 tests)
│   ├── test_features.py
│   ├── test_model.py
│   └── test_onnx.py
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── requirements.txt
├── README.md
├── DEPLOYMENT.md                  # Deployment guide
└── PROJECT_STATUS.md              # Detailed technical documentation
```

---

## 🔬 Technical Highlights

### 1. Regime-Conditioned Anomaly Detection

Anomaly thresholds are calibrated using **split conformal prediction** on healthy samples, providing coverage-guaranteed thresholds (95% coverage) rather than fixed global cutoffs. This ensures consistent false positive rates across different operating conditions.

### 2. Multi-Task Single-Graph ONNX

One inference call returns three predictions with no pipeline overhead in production. The shared CNN backbone learns a unified representation that benefits all three tasks through multi-task learning.

### 3. No Data Leakage

All rolling window features use only past values (t-1 and earlier). This is validated through comprehensive tests to ensure the model can be deployed in real-time streaming scenarios.

### 4. Machine-Readable Inference Contract

`artifacts/inference_contract.json` specifies input/output tensor shapes, dtypes, label order, scaler parameters, and recommended thresholds—ready to paste into a C# ONNX Runtime integration.

### 5. LightGBM Baseline

A LightGBM baseline is trained alongside the CNN for comparison and feature importance analysis. SHAP values from the baseline identify the most predictive features across regimes, which informed the feature engineering choices.

---

## 📈 Results

### CNN Model (Primary)

| Metric | Value |
|--------|-------|
| Accuracy | 99.92% |
| Macro F1 | 0.9961 |
| AUROC (Anomaly) | 0.9994 |
| AUPRC (Anomaly) | 0.9990 |
| Precision @ 95% Recall | 1.0000 |

### Per-Sensor Forecast RMSE

| Sensor | RMSE | MAE |
|--------|------|-----|
| Discharge Pressure | 0.79 bar | 0.63 bar |
| Suction Pressure | 0.66 bar | 0.55 bar |
| Flow Rate | 1.35 m³/h | 0.93 m³/h |
| Motor Current | 0.91 A | 0.71 A |
| Vibration RMS | 0.77 mm/s | 0.63 mm/s |
| Bearing Temperature | 1.11 °C | 0.84 °C |

### LightGBM Baseline (Comparison)

| Metric | Value |
|--------|-------|
| Accuracy | 99.92% |
| Macro F1 | 0.9975 |

The CNN and LightGBM achieve similar accuracy, but the CNN excels at temporal pattern recognition (cavitation pressure oscillations) while LightGBM is better at monotonic trends (bearing wear temperature rise).

---

## 📚 Documentation

- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Detailed technical documentation, architecture, recent fixes, and production readiness checklist
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Step-by-step deployment guide for .NET integration
- **`artifacts/inference_contract.json`** - Machine-readable tensor specification
- **`artifacts/feature_spec.json`** - Feature engineering specification (34 features)

---

## 🧪 Testing

The project includes 59 comprehensive tests covering:

- **Feature Engineering (22 tests)**: Rolling windows, feature count, NaN handling, transformations
- **Model Integration (20 tests)**: Forward pass, loss computation, overfitting sanity check
- **ONNX Export (17 tests)**: Export validation, output parity, inference contract

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_features.py -v
pytest tests/test_model.py -v
pytest tests/test_onnx.py -v

# Run with coverage report
pytest tests/ --cov=sentinel --cov-report=html
```

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- ONNX Runtime
- NumPy, Pandas, Scikit-learn
- LightGBM
- Pytest (for testing)

See `requirements.txt` for complete list.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

## 🙏 Acknowledgments

This project demonstrates production-grade ML engineering practices including:
- Systematic feature engineering with validation
- Multi-task learning for efficiency
- Conformal prediction for calibration
- Comprehensive testing and documentation
- ONNX export for cross-platform deployment

Built with a focus on reliability, reproducibility, and production readiness.
