"""
Standalone evaluation script for IndustrialSentinel.

Usage
-----
    python evaluate.py                                                    # uses defaults
    python evaluate.py --checkpoint artifacts/sentinel_best.pt            # specify checkpoint
    python evaluate.py --data data/test.csv                               # use CSV data
    python evaluate.py --data simulate --sim_hours 50.0                   # generate new data
    python evaluate.py --artifacts_dir results                            # custom output dir

This script loads a trained model checkpoint and evaluates it on a specified dataset,
computing comprehensive metrics across all three tasks (regime classification, anomaly
detection, and forecasting).
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score

from sentinel.datagen.pump_sim import PumpConfig, simulate, SENSOR_COLS
from sentinel.features.pipeline import FeaturePipeline
from sentinel.models.cnn import SentinelCNN, FORECAST_STEPS


# ---------------------------------------------------------------------------
# Dataset (same as train.py)
# ---------------------------------------------------------------------------
class SensorDataset(Dataset):
    """
    Sliding-window dataset from a feature matrix.

    Each sample = window of (seq_len) steps.
    Labels:
      - regime_label  : int class for classification
      - anomaly_label : 1.0 if NOT healthy, else 0.0
      - forecast      : next FORECAST_STEPS raw sensor values (shape: steps × n_sensors)
    """

    def __init__(
        self,
        X: np.ndarray,           # (N, n_features)  scaled features
        regime_labels: np.ndarray,  # (N,) int
        raw_sensors: np.ndarray, # (N, n_raw_sensors) unscaled raw values for forecast targets
        seq_len: int = 60,
    ):
        self.X      = torch.from_numpy(X.astype(np.float32))
        self.regime = torch.from_numpy(regime_labels.astype(np.int64))
        self.raw    = torch.from_numpy(raw_sensors.astype(np.float32))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len - FORECAST_STEPS

    def __getitem__(self, idx):
        x       = self.X[idx: idx + self.seq_len]                    # (seq_len, n_feat)
        regime  = self.regime[idx + self.seq_len - 1]                # scalar
        anomaly = (regime > 0).float()                               # 1 if fault
        forecast = self.raw[idx + self.seq_len:
                            idx + self.seq_len + FORECAST_STEPS]     # (steps, n_raw)
        return x, regime, anomaly, forecast


# ---------------------------------------------------------------------------
# Metrics computation (reused from train.py)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_comprehensive_metrics(model, loader, device):
    """
    Compute comprehensive evaluation metrics across all three tasks.
    
    Returns
    -------
    dict with keys:
        - regime_classification: macro F1, per-class precision/recall, confusion matrix
        - anomaly_detection: AUROC, AUPRC, precision @ 95% recall
        - forecasting: per-sensor RMSE, per-sensor MAE
    """
    model.eval()
    
    # Collect all predictions and targets
    all_regime_preds = []
    all_regime_targets = []
    all_anomaly_preds = []
    all_anomaly_targets = []
    all_forecast_preds = []
    all_forecast_targets = []
    
    for x, reg, anom, fcast in loader:
        x = x.to(device)
        logits, anom_pred, fcast_pred = model(x)
        
        # Regime classification
        regime_pred = logits.argmax(1).cpu().numpy()
        all_regime_preds.append(regime_pred)
        all_regime_targets.append(reg.numpy())
        
        # Anomaly detection
        all_anomaly_preds.append(anom_pred.squeeze(-1).cpu().numpy())
        all_anomaly_targets.append(anom.numpy())
        
        # Forecasting
        all_forecast_preds.append(fcast_pred.cpu().numpy())
        all_forecast_targets.append(fcast.numpy())
    
    # Concatenate all batches
    regime_preds = np.concatenate(all_regime_preds)
    regime_targets = np.concatenate(all_regime_targets)
    anomaly_preds = np.concatenate(all_anomaly_preds)
    anomaly_targets = np.concatenate(all_anomaly_targets)
    forecast_preds = np.concatenate(all_forecast_preds)
    forecast_targets = np.concatenate(all_forecast_targets)
    
    # ===== Regime Classification Metrics =====
    macro_f1 = f1_score(regime_targets, regime_preds, average='macro')
    per_class_precision = precision_score(regime_targets, regime_preds, average=None)
    per_class_recall = recall_score(regime_targets, regime_preds, average=None)
    conf_matrix = confusion_matrix(regime_targets, regime_preds)
    
    # Overall accuracy
    accuracy = (regime_preds == regime_targets).mean()
    
    # ===== Anomaly Detection Metrics =====
    auroc = roc_auc_score(anomaly_targets, anomaly_preds)
    auprc = average_precision_score(anomaly_targets, anomaly_preds)
    
    # Precision @ 95% recall
    # Sort by anomaly score descending
    sorted_indices = np.argsort(-anomaly_preds)
    sorted_targets = anomaly_targets[sorted_indices]
    
    # Find threshold that gives 95% recall
    n_positives = sorted_targets.sum()
    n_to_retrieve = int(np.ceil(0.95 * n_positives))
    
    if n_to_retrieve > 0 and n_positives > 0:
        # Find how many samples we need to retrieve to get 95% of positives
        cumsum = np.cumsum(sorted_targets)
        idx_95_recall = np.where(cumsum >= n_to_retrieve)[0]
        if len(idx_95_recall) > 0:
            idx_95_recall = idx_95_recall[0]
            precision_at_95_recall = sorted_targets[:idx_95_recall + 1].sum() / (idx_95_recall + 1)
        else:
            precision_at_95_recall = 0.0
    else:
        precision_at_95_recall = 0.0
    
    # ===== Forecasting Metrics =====
    # forecast_preds shape: (N, forecast_steps, n_sensors)
    # forecast_targets shape: (N, forecast_steps, n_sensors)
    
    # Compute per-sensor RMSE and MAE
    n_sensors = forecast_preds.shape[2]
    per_sensor_rmse = []
    per_sensor_mae = []
    
    for sensor_idx in range(n_sensors):
        sensor_preds = forecast_preds[:, :, sensor_idx]
        sensor_targets = forecast_targets[:, :, sensor_idx]
        
        # RMSE
        mse = np.mean((sensor_preds - sensor_targets) ** 2)
        rmse = np.sqrt(mse)
        per_sensor_rmse.append(float(rmse))
        
        # MAE
        mae = np.mean(np.abs(sensor_preds - sensor_targets))
        per_sensor_mae.append(float(mae))
    
    # Build metrics dictionary
    metrics = {
        "regime_classification": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "per_class_precision": per_class_precision.tolist(),
            "per_class_recall": per_class_recall.tolist(),
            "confusion_matrix": conf_matrix.tolist(),
        },
        "anomaly_detection": {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "precision_at_95_recall": float(precision_at_95_recall),
        },
        "forecasting": {
            "per_sensor_rmse": per_sensor_rmse,
            "per_sensor_mae": per_sensor_mae,
        }
    }
    
    return metrics


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create artifacts directory if it doesn't exist
    os.makedirs(args.artifacts_dir, exist_ok=True)
    
    # 1. Load feature pipeline
    print("\nLoading feature pipeline...")
    pipeline_path = os.path.join(os.path.dirname(args.checkpoint), "pipeline.pkl")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Feature pipeline not found at {pipeline_path}. "
            f"Make sure pipeline.pkl exists in the same directory as the checkpoint."
        )
    
    with open(pipeline_path, "rb") as f:
        pipeline = pickle.load(f)
    print(f"  Loaded pipeline from {pipeline_path}")
    
    # 2. Load or generate data
    print("\nLoading data...")
    if args.data == "simulate":
        print(f"  Generating {args.sim_hours} hours of simulation data...")
        sim_cfg = PumpConfig(duration_hours=args.sim_hours, seed=42, sample_rate_hz=1.0)
        df = simulate(sim_cfg)
        print(f"  Generated {len(df):,} samples")
    else:
        print(f"  Loading data from {args.data}...")
        df = pd.read_csv(args.data)
        print(f"  Loaded {len(df):,} samples")
    
    # Verify required columns exist
    required_cols = SENSOR_COLS + ["regime_label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")
    
    print(f"  Regime distribution:\n{df['regime_label'].value_counts().sort_index()}")
    
    # 3. Apply feature pipeline transformation
    print("\nApplying feature pipeline transformation...")
    X = pipeline.transform(df)
    regime_labels = df["regime_label"].values[len(df) - len(X):]  # align after warmup drop
    raw_sensors = df[SENSOR_COLS].values[len(df) - len(X):]
    print(f"  Feature matrix: {X.shape}")
    
    # 4. Load run metadata to get seq_len and n_features
    print("\nLoading model configuration...")
    run_meta_path = os.path.join(os.path.dirname(args.checkpoint), "run_meta.json")
    if not os.path.exists(run_meta_path):
        raise FileNotFoundError(
            f"Run metadata not found at {run_meta_path}. "
            f"Make sure run_meta.json exists in the same directory as the checkpoint."
        )
    
    with open(run_meta_path, "r") as f:
        run_meta = json.load(f)
    
    seq_len = run_meta["seq_len"]
    n_features = run_meta["n_features"]
    print(f"  seq_len: {seq_len}, n_features: {n_features}")
    
    # 5. Create DataLoader
    print("\nCreating DataLoader...")
    dataset = SensorDataset(X, regime_labels, raw_sensors, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    print(f"  Dataset size: {len(dataset)} samples")
    
    # 6. Load trained model
    print(f"\nLoading model from {args.checkpoint}...")
    model = SentinelCNN(n_features=n_features, seq_len=seq_len, dropout=0.3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("  Model loaded successfully")
    
    # 7. Compute comprehensive metrics
    print("\nComputing comprehensive metrics...")
    metrics = compute_comprehensive_metrics(model, loader, device)
    
    # 8. Print summary to console
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\n--- Regime Classification ---")
    print(f"  Accuracy:            {metrics['regime_classification']['accuracy']:.4f}")
    print(f"  Macro F1:            {metrics['regime_classification']['macro_f1']:.4f}")
    print(f"  Per-class Precision: {[f'{p:.4f}' for p in metrics['regime_classification']['per_class_precision']]}")
    print(f"  Per-class Recall:    {[f'{r:.4f}' for r in metrics['regime_classification']['per_class_recall']]}")
    print(f"  Confusion Matrix:")
    for row in metrics['regime_classification']['confusion_matrix']:
        print(f"    {row}")
    
    print("\n--- Anomaly Detection ---")
    print(f"  AUROC:                  {metrics['anomaly_detection']['auroc']:.4f}")
    print(f"  AUPRC:                  {metrics['anomaly_detection']['auprc']:.4f}")
    print(f"  Precision @ 95% Recall: {metrics['anomaly_detection']['precision_at_95_recall']:.4f}")
    
    print("\n--- Forecasting ---")
    print(f"  Per-sensor RMSE: {[f'{rmse:.4f}' for rmse in metrics['forecasting']['per_sensor_rmse']]}")
    print(f"  Per-sensor MAE:  {[f'{mae:.4f}' for mae in metrics['forecasting']['per_sensor_mae']]}")
    print("="*60 + "\n")
    
    # 9. Save metrics to JSON
    output_path = os.path.join(args.artifacts_dir, "eval_metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained IndustrialSentinel model on a dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/sentinel_best.pt",
        help="Path to trained model checkpoint (default: artifacts/sentinel_best.pt)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="simulate",
        help='Path to CSV data file or "simulate" to generate new data (default: simulate)'
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default="artifacts",
        help="Directory to save evaluation results (default: artifacts)"
    )
    parser.add_argument(
        "--sim_hours",
        type=float,
        default=50.0,
        help="Hours of simulation data if using 'simulate' (default: 50.0)"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    main(args)
