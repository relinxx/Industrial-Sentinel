# Deployment Guide

This guide provides step-by-step instructions for training, evaluating, exporting, and deploying the Industrial-Sentinel model in a .NET/C# production environment.

---

## Table of Contents

1. [Training the Model](#1-training-the-model)
2. [Evaluating Performance](#2-evaluating-performance)
3. [Exporting to ONNX](#3-exporting-to-onnx)
4. [Integrating with .NET/C#](#4-integrating-with-netc)
5. [Production Deployment Checklist](#5-production-deployment-checklist)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Training the Model

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster training)
- 8GB+ RAM
- 2GB+ disk space

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/industrial-sentinel.git
cd industrial-sentinel

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training Options

#### Option A: Full Training (Recommended for Production)

```bash
python train.py
```

**Configuration:**
- Simulation: 200 hours of sensor data
- Training time: ~20 minutes on RTX 4050 GPU, ~45 minutes on CPU
- Dataset size: 720,000 samples
- Output: All artifacts saved to `artifacts/`

**Expected Output:**
```
✅ Training complete!
   Accuracy: 99.92%
   Macro F1: 0.9961
   AUROC: 0.9994
   
   Artifacts saved to artifacts/
```

#### Option B: Quick Training (For Testing)

```bash
python train.py --sim_hours 10 --epochs 20
```

**Configuration:**
- Simulation: 10 hours of sensor data
- Training time: ~5 minutes
- Dataset size: 36,000 samples
- ⚠️ Warning: Short runs may not include all 4 fault regimes

#### Option C: Custom Configuration

```bash
python train.py \
  --sim_hours 50 \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-3 \
  --device cuda
```

**Available Parameters:**
- `--sim_hours`: Hours of sensor data to simulate (default: 200)
- `--epochs`: Maximum training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 3e-4)
- `--device`: Device to use (cuda/cpu, default: auto-detect)
- `--config`: Path to YAML config file (default: configs/train.yaml)

### Training Configuration File

Edit `configs/train.yaml` to customize training:

```yaml
# Data generation
sim_hours: 200
sample_rate: 1.0  # Hz

# Model architecture
hidden_dim: 128
num_conv_layers: 4

# Training
batch_size: 64
learning_rate: 0.0003
max_epochs: 50
early_stopping_patience: 8

# Loss weights
loss_weights:
  w_regime: 1.0      # Regime classification
  w_anomaly: 0.5     # Anomaly detection
  w_forecast: 0.05   # Forecasting

# Optimizer
optimizer:
  type: AdamW
  weight_decay: 0.0001

# Scheduler
scheduler:
  type: CosineAnnealingLR
  T_max: 50
```

### Generated Artifacts

After training, the following files are created in `artifacts/`:

| File | Size | Purpose |
|------|------|---------|
| `sentinel_best.pt` | ~570 KB | PyTorch model checkpoint |
| `pipeline.pkl` | ~50 KB | Feature engineering pipeline |
| `lgbm_baseline.pkl` | ~200 KB | LightGBM baseline model |
| `test_metrics.json` | ~2 KB | Comprehensive evaluation metrics |
| `run_meta.json` | ~1 KB | Training run metadata |
| `scaler_config.json` | ~5 KB | StandardScaler parameters |
| `feature_spec.json` | ~10 KB | Feature engineering specification |

---

## 2. Evaluating Performance

### Standalone Evaluation

```bash
# Evaluate on simulated data
python evaluate.py \
  --checkpoint artifacts/sentinel_best.pt \
  --data simulate \
  --sim_hours 50

# Evaluate on CSV data
python evaluate.py \
  --checkpoint artifacts/sentinel_best.pt \
  --data path/to/sensor_data.csv

# Save to custom directory
python evaluate.py \
  --checkpoint artifacts/sentinel_best.pt \
  --data simulate \
  --artifacts_dir results/
```

### CSV Data Format

If evaluating on real data, provide a CSV with these columns:

```csv
timestamp,discharge_pressure,suction_pressure,flow_rate,motor_current,vibration_rms,bearing_temp,regime,is_anomaly
2024-01-01 00:00:00,8.5,2.1,100.0,45.0,2.5,65.0,0,0
2024-01-01 00:00:01,8.4,2.0,99.5,44.8,2.4,65.1,0,0
...
```

**Column Specifications:**
- `timestamp`: ISO 8601 format
- Sensor columns: Float values in specified units
- `regime`: Integer (0=HEALTHY, 1=CAVITATION, 2=BEARING_WEAR, 3=SEAL_LEAK)
- `is_anomaly`: Binary (0=normal, 1=anomalous)

### Understanding Evaluation Metrics

**Regime Classification:**
- **Accuracy**: Overall correct predictions (target: >99%)
- **Macro F1**: Average F1 across all regimes (target: >0.99)
- **Per-class Precision/Recall**: Performance per regime

**Anomaly Detection:**
- **AUROC**: Area under ROC curve (target: >0.99)
- **AUPRC**: Area under precision-recall curve (target: >0.99)
- **Precision @ 95% Recall**: Precision when catching 95% of anomalies (target: >0.95)

**Forecasting:**
- **RMSE**: Root mean squared error per sensor (lower is better)
- **MAE**: Mean absolute error per sensor (lower is better)

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_features.py -v      # Feature engineering tests
pytest tests/test_model.py -v         # Model integration tests
pytest tests/test_onnx.py -v          # ONNX export tests

# Run with coverage report
pytest tests/ --cov=sentinel --cov-report=html
open htmlcov/index.html  # View coverage report
```

---

## 3. Exporting to ONNX

### Export Process

```bash
python -m sentinel.export.onnx_export
```

**Generated Files:**
- `artifacts/sentinel.onnx` (39.3 KB) - ONNX model
- `artifacts/sentinel.onnx.data` - External tensor data (if model >2GB)
- `artifacts/inference_contract.json` - Tensor specifications for C# integration

### Verifying ONNX Export

The export script automatically validates:
1. ✅ ONNX model loads successfully
2. ✅ Output shapes match expected dimensions
3. ✅ Output values match PyTorch model (parity check)
4. ✅ Inference contract is valid JSON

**Expected Output:**
```
✅ ONNX export successful!
   Model: artifacts/sentinel.onnx (39.3 KB)
   Opset: 17
   Inputs: sensor_window (1, 60, 34)
   Outputs:
     - regime_logits (1, 4)
     - anomaly_score (1, 1)
     - forecast (1, 10, 6)
   
   Parity check: PASSED
   Max difference: 1.2e-6
```

### Calibrating Anomaly Thresholds

```bash
python -m sentinel.calibrate.conformal
```

**Generated File:**
- `artifacts/thresholds.json` - Calibrated anomaly threshold

**Calibration Method:**
- Uses split conformal prediction on healthy samples
- Provides 95% coverage guarantee
- Single global threshold (not per-regime)

**Expected Output:**
```
✅ Conformal calibration complete!
   Coverage level: 95%
   Calibrated threshold: 0.007432
   
   Saved to artifacts/thresholds.json
```

---

## 4. Integrating with .NET/C#

### Step 1: Install ONNX Runtime

```bash
dotnet add package Microsoft.ML.OnnxRuntime
```

Or add to your `.csproj`:

```xml
<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.16.0" />
```

### Step 2: Load Inference Contract

```csharp
using System.Text.Json;

// Load inference contract
var contractJson = File.ReadAllText("artifacts/inference_contract.json");
var contract = JsonSerializer.Deserialize<InferenceContract>(contractJson);

// Load thresholds
var thresholdsJson = File.ReadAllText("artifacts/thresholds.json");
var thresholds = JsonSerializer.Deserialize<Thresholds>(thresholdsJson);

// Load scaler config
var scalerJson = File.ReadAllText("artifacts/scaler_config.json");
var scaler = JsonSerializer.Deserialize<ScalerConfig>(scalerJson);
```

**Contract Classes:**

```csharp
public class InferenceContract
{
    public InputSpec Input { get; set; }
    public Dictionary<string, OutputSpec> Outputs { get; set; }
    public string[] RegimeLabels { get; set; }
    public string[] SensorNames { get; set; }
}

public class InputSpec
{
    public string Name { get; set; }
    public int[] Shape { get; set; }  // [1, 60, 34]
    public string Dtype { get; set; }  // "float32"
}

public class OutputSpec
{
    public string Name { get; set; }
    public int[] Shape { get; set; }
    public string Dtype { get; set; }
}

public class Thresholds
{
    public double AnomalyThreshold { get; set; }
}

public class ScalerConfig
{
    public double[] Mean { get; set; }  // Length 34
    public double[] Std { get; set; }   // Length 34
}
```

### Step 3: Implement Feature Engineering

```csharp
using System;
using System.Linq;

public class FeatureEngineer
{
    private readonly ScalerConfig _scaler;
    private readonly Queue<SensorReading> _buffer;
    
    public FeatureEngineer(ScalerConfig scaler)
    {
        _scaler = scaler;
        _buffer = new Queue<SensorReading>(60);
    }
    
    public void AddReading(SensorReading reading)
    {
        _buffer.Enqueue(reading);
        if (_buffer.Count > 60)
            _buffer.Dequeue();
    }
    
    public float[] ExtractFeatures()
    {
        if (_buffer.Count < 60)
            throw new InvalidOperationException("Need 60 readings for inference");
        
        var readings = _buffer.ToArray();
        var features = new List<float>();
        
        // For each timestep, compute 34 features
        for (int t = 0; t < 60; t++)
        {
            var current = readings[t];
            
            // 1. Raw sensor values (6 features)
            features.Add(current.DischargePressure);
            features.Add(current.SuctionPressure);
            features.Add(current.FlowRate);
            features.Add(current.MotorCurrent);
            features.Add(current.VibrationRms);
            features.Add(current.BearingTemp);
            
            // 2. Rolling mean (30-step window, 6 features)
            if (t >= 29)
            {
                features.Add(RollingMean(readings, t, 30, r => r.DischargePressure));
                features.Add(RollingMean(readings, t, 30, r => r.SuctionPressure));
                features.Add(RollingMean(readings, t, 30, r => r.FlowRate));
                features.Add(RollingMean(readings, t, 30, r => r.MotorCurrent));
                features.Add(RollingMean(readings, t, 30, r => r.VibrationRms));
                features.Add(RollingMean(readings, t, 30, r => r.BearingTemp));
            }
            else
            {
                features.AddRange(Enumerable.Repeat(0f, 6));
            }
            
            // 3. Rolling std (30-step window, 6 features)
            if (t >= 29)
            {
                features.Add(RollingStd(readings, t, 30, r => r.DischargePressure));
                features.Add(RollingStd(readings, t, 30, r => r.SuctionPressure));
                features.Add(RollingStd(readings, t, 30, r => r.FlowRate));
                features.Add(RollingStd(readings, t, 30, r => r.MotorCurrent));
                features.Add(RollingStd(readings, t, 30, r => r.VibrationRms));
                features.Add(RollingStd(readings, t, 30, r => r.BearingTemp));
            }
            else
            {
                features.AddRange(Enumerable.Repeat(0f, 6));
            }
            
            // 4. Slope (20-step OLS, 6 features)
            if (t >= 19)
            {
                features.Add(RollingSlope(readings, t, 20, r => r.DischargePressure));
                features.Add(RollingSlope(readings, t, 20, r => r.SuctionPressure));
                features.Add(RollingSlope(readings, t, 20, r => r.FlowRate));
                features.Add(RollingSlope(readings, t, 20, r => r.MotorCurrent));
                features.Add(RollingSlope(readings, t, 20, r => r.VibrationRms));
                features.Add(RollingSlope(readings, t, 20, r => r.BearingTemp));
            }
            else
            {
                features.AddRange(Enumerable.Repeat(0f, 6));
            }
            
            // 5. Delta (first difference, 6 features)
            if (t > 0)
            {
                var prev = readings[t - 1];
                features.Add(current.DischargePressure - prev.DischargePressure);
                features.Add(current.SuctionPressure - prev.SuctionPressure);
                features.Add(current.FlowRate - prev.FlowRate);
                features.Add(current.MotorCurrent - prev.MotorCurrent);
                features.Add(current.VibrationRms - prev.VibrationRms);
                features.Add(current.BearingTemp - prev.BearingTemp);
            }
            else
            {
                features.AddRange(Enumerable.Repeat(0f, 6));
            }
            
            // 6. Cross-sensor ratios (2 features)
            features.Add(current.DischargePressure / (current.SuctionPressure + 1e-6f));
            features.Add(current.FlowRate / (current.MotorCurrent + 1e-6f));
            
            // 7. FFT features (2 features) - simplified, compute on vibration window
            if (t >= 63)
            {
                var (peakFreq, peakAmp) = ComputeFFT(readings, t);
                features.Add(peakFreq);
                features.Add(peakAmp);
            }
            else
            {
                features.Add(0f);
                features.Add(0f);
            }
        }
        
        // Apply StandardScaler: (x - mean) / std
        for (int i = 0; i < features.Count; i++)
        {
            features[i] = (features[i] - (float)_scaler.Mean[i % 34]) / (float)_scaler.Std[i % 34];
        }
        
        return features.ToArray();
    }
    
    private float RollingMean(SensorReading[] readings, int endIdx, int window, Func<SensorReading, float> selector)
    {
        var sum = 0f;
        for (int i = endIdx - window + 1; i <= endIdx; i++)
            sum += selector(readings[i]);
        return sum / window;
    }
    
    private float RollingStd(SensorReading[] readings, int endIdx, int window, Func<SensorReading, float> selector)
    {
        var mean = RollingMean(readings, endIdx, window, selector);
        var sumSq = 0f;
        for (int i = endIdx - window + 1; i <= endIdx; i++)
        {
            var diff = selector(readings[i]) - mean;
            sumSq += diff * diff;
        }
        return (float)Math.Sqrt(sumSq / window);
    }
    
    private float RollingSlope(SensorReading[] readings, int endIdx, int window, Func<SensorReading, float> selector)
    {
        // Simple OLS: slope = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
        var xMean = (window - 1) / 2f;
        var yMean = RollingMean(readings, endIdx, window, selector);
        
        var numerator = 0f;
        var denominator = 0f;
        for (int i = 0; i < window; i++)
        {
            var x = i;
            var y = selector(readings[endIdx - window + 1 + i]);
            numerator += (x - xMean) * (y - yMean);
            denominator += (x - xMean) * (x - xMean);
        }
        
        return numerator / (denominator + 1e-6f);
    }
    
    private (float peakFreq, float peakAmp) ComputeFFT(SensorReading[] readings, int endIdx)
    {
        // Simplified FFT on 64-point vibration window
        // Use a proper FFT library like MathNet.Numerics in production
        var window = 64;
        var vibration = new float[window];
        for (int i = 0; i < window; i++)
            vibration[i] = readings[endIdx - window + 1 + i].VibrationRms;
        
        // Placeholder: return dummy values
        // In production, use MathNet.Numerics.IntegralTransforms.Fourier.Forward()
        return (0f, 0f);
    }
}

public class SensorReading
{
    public DateTime Timestamp { get; set; }
    public float DischargePressure { get; set; }
    public float SuctionPressure { get; set; }
    public float FlowRate { get; set; }
    public float MotorCurrent { get; set; }
    public float VibrationRms { get; set; }
    public float BearingTemp { get; set; }
}
```

### Step 4: Run ONNX Inference

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class SentinelInference
{
    private readonly InferenceSession _session;
    private readonly FeatureEngineer _featureEngineer;
    private readonly Thresholds _thresholds;
    private readonly string[] _regimeLabels;
    
    public SentinelInference(string modelPath, ScalerConfig scaler, Thresholds thresholds, string[] regimeLabels)
    {
        _session = new InferenceSession(modelPath);
        _featureEngineer = new FeatureEngineer(scaler);
        _thresholds = thresholds;
        _regimeLabels = regimeLabels;
    }
    
    public void AddReading(SensorReading reading)
    {
        _featureEngineer.AddReading(reading);
    }
    
    public PredictionResult Predict()
    {
        // Extract features (1, 60, 34)
        var features = _featureEngineer.ExtractFeatures();
        var tensor = new DenseTensor<float>(features, new[] { 1, 60, 34 });
        
        // Create input
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("sensor_window", tensor)
        };
        
        // Run inference
        using var results = _session.Run(inputs);
        
        // Extract outputs
        var regimeLogits = results.First(r => r.Name == "regime_logits").AsTensor<float>().ToArray();
        var anomalyScore = results.First(r => r.Name == "anomaly_score").AsTensor<float>().ToArray()[0];
        var forecast = results.First(r => r.Name == "forecast").AsTensor<float>().ToArray();
        
        // Apply softmax to regime logits
        var regimeProbs = Softmax(regimeLogits);
        var detectedRegime = Array.IndexOf(regimeProbs, regimeProbs.Max());
        
        // Apply threshold to anomaly score
        var isAnomalous = anomalyScore > _thresholds.AnomalyThreshold;
        
        return new PredictionResult
        {
            DetectedRegime = _regimeLabels[detectedRegime],
            RegimeProbabilities = regimeProbs,
            AnomalyScore = anomalyScore,
            IsAnomalous = isAnomalous,
            Forecast = forecast
        };
    }
    
    private float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var exp = logits.Select(x => Math.Exp(x - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(x => (float)(x / sum)).ToArray();
    }
}

public class PredictionResult
{
    public string DetectedRegime { get; set; }
    public float[] RegimeProbabilities { get; set; }
    public float AnomalyScore { get; set; }
    public bool IsAnomalous { get; set; }
    public float[] Forecast { get; set; }  // Shape: (10, 6) flattened
}
```

### Step 5: Usage Example

```csharp
// Initialize
var contract = LoadInferenceContract("artifacts/inference_contract.json");
var scaler = LoadScalerConfig("artifacts/scaler_config.json");
var thresholds = LoadThresholds("artifacts/thresholds.json");

var inference = new SentinelInference(
    "artifacts/sentinel.onnx",
    scaler,
    thresholds,
    contract.RegimeLabels
);

// Stream sensor readings
while (true)
{
    var reading = GetNextSensorReading();  // Your data source
    inference.AddReading(reading);
    
    // Run inference every second (after 60 readings buffered)
    if (ReadyForInference())
    {
        var result = inference.Predict();
        
        Console.WriteLine($"Regime: {result.DetectedRegime}");
        Console.WriteLine($"Anomaly Score: {result.AnomalyScore:F4}");
        Console.WriteLine($"Is Anomalous: {result.IsAnomalous}");
        
        if (result.IsAnomalous)
        {
            TriggerAlert(result);
        }
    }
}
```

---

## 5. Production Deployment Checklist

### Pre-Deployment

- [ ] **Model Training**
  - [ ] Trained on sufficient data (200+ hours recommended)
  - [ ] Validation metrics meet requirements (>99% accuracy, >0.99 AUROC)
  - [ ] All 4 regimes represented in training data
  - [ ] Test suite passes (59 tests)

- [ ] **ONNX Export**
  - [ ] ONNX model exported successfully
  - [ ] Parity check passed (PyTorch vs ONNX outputs match)
  - [ ] Inference contract generated
  - [ ] Thresholds calibrated

- [ ] **Feature Engineering**
  - [ ] C# feature engineering implemented
  - [ ] StandardScaler applied correctly
  - [ ] Rolling window features validated
  - [ ] FFT features implemented (if needed)

- [ ] **Integration Testing**
  - [ ] ONNX Runtime loads model successfully
  - [ ] Input tensor shape correct (1, 60, 34)
  - [ ] Output tensors extracted correctly
  - [ ] Threshold application works
  - [ ] End-to-end inference pipeline tested

### Deployment

- [ ] **Infrastructure**
  - [ ] ONNX Runtime installed (.NET package)
  - [ ] Model files deployed to production
  - [ ] Configuration files deployed (scaler, thresholds, contract)
  - [ ] Logging configured
  - [ ] Monitoring configured

- [ ] **Data Pipeline**
  - [ ] Real-time sensor data ingestion working
  - [ ] Data resampling to 1 Hz implemented
  - [ ] Missing data handling (forward-fill)
  - [ ] 60-reading buffer management

- [ ] **Alerting**
  - [ ] Anomaly detection alerts configured
  - [ ] Alert thresholds set correctly
  - [ ] Notification system integrated
  - [ ] Alert fatigue mitigation (rate limiting, deduplication)

### Post-Deployment

- [ ] **Monitoring**
  - [ ] Inference latency monitored
  - [ ] Prediction distribution monitored
  - [ ] Anomaly rate monitored
  - [ ] Model drift detection configured

- [ ] **Validation**
  - [ ] Compare predictions to ground truth (if available)
  - [ ] Validate regime detection accuracy
  - [ ] Validate anomaly detection precision/recall
  - [ ] Validate forecast accuracy

- [ ] **Maintenance**
  - [ ] Model retraining schedule defined
  - [ ] Data collection for retraining
  - [ ] Version control for models
  - [ ] Rollback procedure defined

---

## 6. Troubleshooting

### Training Issues

**Problem: Training loss starts very high (>30)**

**Solution:** Check forecast loss weight in `configs/train.yaml`. Should be `w_forecast: 0.05` for raw sensor values.

---

**Problem: Short runs don't include all 4 regimes**

**Solution:** Increase `sim_hours` to 50+ for reliable evaluation, 200+ for production metrics.

---

**Problem: GPU out of memory**

**Solution:** Reduce `batch_size` in `configs/train.yaml` or use CPU with `--device cpu`.

---

### ONNX Export Issues

**Problem: ONNX export fails with "Unsupported operator"**

**Solution:** Ensure PyTorch version is compatible with ONNX opset 17. Update PyTorch: `pip install --upgrade torch`.

---

**Problem: Parity check fails (outputs don't match)**

**Solution:** This indicates a bug in the export. Check for:
- Dynamic axes in ONNX export
- Unsupported operations (e.g., in-place operations)
- Incorrect input shapes

---

### C# Integration Issues

**Problem: ONNX Runtime throws "Invalid input shape"**

**Solution:** Verify input tensor shape is exactly `[1, 60, 34]`. Check feature engineering produces 34 features per timestep.

---

**Problem: Feature engineering produces NaN values**

**Solution:** Check for:
- Division by zero (add epsilon: `1e-6`)
- Insufficient buffer size for rolling windows
- Missing data handling (forward-fill)

---

**Problem: Anomaly detection always returns false**

**Solution:** Verify threshold is loaded correctly from `thresholds.json`. Check anomaly score is not being clipped or normalized incorrectly.

---

### Performance Issues

**Problem: Inference latency too high (>50ms)**

**Solution:**
- Use ONNX Runtime GPU provider
- Batch multiple inferences together
- Optimize feature engineering (cache rolling statistics)
- Profile C# code to find bottlenecks

---

**Problem: High memory usage**

**Solution:**
- Dispose ONNX Runtime sessions properly (`using` statement)
- Limit buffer size to exactly 60 readings
- Clear old predictions from memory

---

### Accuracy Issues

**Problem: Regime detection accuracy lower than expected in production**

**Solution:**
- Verify real sensor data matches simulation characteristics
- Check for data quality issues (missing values, outliers)
- Retrain model on real data if available
- Validate feature engineering matches Python implementation

---

**Problem: High false positive rate for anomaly detection**

**Solution:**
- Recalibrate thresholds on production data
- Adjust coverage level (e.g., 99% instead of 95%)
- Implement per-regime thresholds if needed
- Add temporal smoothing (e.g., require 3 consecutive anomalies)

---

## Additional Resources

- **[README.md](README.md)** - Project overview and quick start
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Detailed technical documentation
- **`artifacts/inference_contract.json`** - Machine-readable tensor specification
- **`artifacts/feature_spec.json`** - Feature engineering specification
- **ONNX Runtime Documentation** - https://onnxruntime.ai/docs/
- **Microsoft.ML.OnnxRuntime NuGet** - https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/

---

## Support

For questions or issues:
1. Check this deployment guide
2. Review [PROJECT_STATUS.md](PROJECT_STATUS.md) for technical details
3. Open an issue on GitHub
4. Contact the development team

---

**Last Updated:** 2024  
**Version:** 1.0
