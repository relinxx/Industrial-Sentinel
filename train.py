"""
Training script for IndustrialSentinel.

Usage
-----
    python train.py                          # uses configs/train.yaml
    python train.py --config configs/train.yaml
    python train.py --epochs 50 --lr 1e-3

Temporal split
--------------
    train : first 70% of time
    val   : next 15%
    test  : final 15%
    (No shuffling across splits — strict temporal ordering.)
"""

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, average_precision_score

from sentinel.datagen.pump_sim import PumpConfig, simulate, SENSOR_COLS
from sentinel.features.pipeline import FeaturePipeline, build_feature_spec
from sentinel.models.cnn import SentinelCNN, SentinelLoss, FORECAST_STEPS, N_RAW_SENSORS
from sentinel.calibrate.conformal import run_calibration
from sentinel.models.lgbm_baseline import train_lgbm, evaluate_lgbm


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DEFAULT_CFG = {
    "seed":          42,
    "seq_len":       60,
    "batch_size":    64,
    "epochs":        40,
    "lr":            3e-4,
    "weight_decay":  1e-4,
    "dropout":       0.3,
    "w_regime":      1.0,
    "w_anomaly":     0.5,
    "w_forecast":    0.05,  # was 0.3
    "patience":      8,
    "sim_hours":     200.0,
    "sim_seed":      42,
    "sample_rate_hz": 1.0,
    "artifacts_dir": "artifacts",
    "data_dir":      "data/sim",
}


# ---------------------------------------------------------------------------
# Dataset
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
# Helpers
# ---------------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def temporal_split(X, regime, raw, fracs=(0.70, 0.15, 0.15)):
    """
    Stratified temporal split that ensures all regimes appear in train/val/test.
    
    For each regime, splits its samples temporally according to fracs,
    then concatenates all regimes' splits together.
    This preserves temporal ordering within each regime while ensuring
    all regimes are represented in each split.
    """
    unique_regimes = np.unique(regime)
    
    train_X, train_regime, train_raw = [], [], []
    val_X, val_regime, val_raw = [], [], []
    test_X, test_regime, test_raw = [], [], []
    
    for reg in unique_regimes:
        # Get indices for this regime
        mask = regime == reg
        indices = np.where(mask)[0]
        
        # Extract data for this regime (maintains temporal order)
        X_reg = X[indices]
        regime_reg = regime[indices]
        raw_reg = raw[indices]
        
        # Split this regime's data temporally
        n = len(X_reg)
        i1 = int(n * fracs[0])
        i2 = int(n * (fracs[0] + fracs[1]))
        
        # Append to respective splits
        train_X.append(X_reg[:i1])
        train_regime.append(regime_reg[:i1])
        train_raw.append(raw_reg[:i1])
        
        val_X.append(X_reg[i1:i2])
        val_regime.append(regime_reg[i1:i2])
        val_raw.append(raw_reg[i1:i2])
        
        test_X.append(X_reg[i2:])
        test_regime.append(regime_reg[i2:])
        test_raw.append(raw_reg[i2:])
    
    # Concatenate all regimes
    return (
        (np.vstack(train_X), np.concatenate(train_regime), np.vstack(train_raw)),
        (np.vstack(val_X), np.concatenate(val_regime), np.vstack(val_raw)),
        (np.vstack(test_X), np.concatenate(test_regime), np.vstack(test_raw)),
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, reg, anom, fcast in loader:
        x, reg, anom, fcast = x.to(device), reg.to(device), anom.to(device), fcast.to(device)
        optimizer.zero_grad()
        logits, anom_pred, fcast_pred = model(x)
        loss, _ = criterion(logits, anom_pred, fcast_pred, reg, anom, fcast)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
        correct += (logits.argmax(1) == reg).sum().item()
        n += len(x)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, reg, anom, fcast in loader:
        x, reg, anom, fcast = x.to(device), reg.to(device), anom.to(device), fcast.to(device)
        logits, anom_pred, fcast_pred = model(x)
        loss, _ = criterion(logits, anom_pred, fcast_pred, reg, anom, fcast)
        total_loss += loss.item() * len(x)
        correct += (logits.argmax(1) == reg).sum().item()
        n += len(x)
    return total_loss / n, correct / n


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
# Main
# ---------------------------------------------------------------------------
def main(cfg: dict):
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg["artifacts_dir"], exist_ok=True)

    # 1. Generate / load data
    print("Generating simulation data...")
    sim_cfg = PumpConfig(duration_hours=cfg["sim_hours"], seed=cfg["sim_seed"],
                         sample_rate_hz=cfg["sample_rate_hz"])
    df = simulate(sim_cfg)
    print(f"  {len(df):,} samples  |  regime counts:\n{df['regime'].value_counts()}")
    
    # Warning for short simulation runs
    if cfg["sim_hours"] < 30:
        print(f"\n⚠️  WARNING: Short simulation run ({cfg['sim_hours']} hours)")
        print("    Runs under 30 hours may not include all 4 fault regimes.")
        print("    Use 50+ hours for reliable evaluation, 200 hours for production metrics.\n")

    # 2. Feature engineering
    print("Engineering features...")
    pipeline = FeaturePipeline(sample_rate_hz=cfg["sample_rate_hz"])
    X, regime_labels = pipeline.fit_transform(df, df["regime_label"])
    raw_sensors = df[SENSOR_COLS].values[
        len(df) - len(X):   # align with feature rows after warmup drop
    ]
    n_features = X.shape[1]
    print(f"  Feature matrix: {X.shape}")

    # Save feature spec and scaler
    spec = build_feature_spec()
    with open(os.path.join(cfg["artifacts_dir"], "feature_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    scaler_cfg = pipeline.scaler_config()
    with open(os.path.join(cfg["artifacts_dir"], "scaler_config.json"), "w") as f:
        json.dump(scaler_cfg, f, indent=2)
    pipeline.save(os.path.join(cfg["artifacts_dir"], "pipeline.pkl"))

    # 3. Temporal split
    (X_tr, r_tr, raw_tr), (X_va, r_va, raw_va), (X_te, r_te, raw_te) = temporal_split(
        X, regime_labels, raw_sensors
    )
    print(f"  Train/Val/Test sizes: {len(X_tr)} / {len(X_va)} / {len(X_te)}")
    print(f"  Train regimes: {np.unique(r_tr, return_counts=True)}")
    print(f"  Val regimes:   {np.unique(r_va, return_counts=True)}")
    print(f"  Test regimes:  {np.unique(r_te, return_counts=True)}")

    # 4. Class weights (imbalanced classes)
    class_weights = compute_class_weights(r_tr, n_classes=4).to(device)
    print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

    # 5. DataLoaders
    seq_len = cfg["seq_len"]
    tr_ds = SensorDataset(X_tr, r_tr, raw_tr, seq_len)
    va_ds = SensorDataset(X_va, r_va, raw_va, seq_len)
    te_ds = SensorDataset(X_te, r_te, raw_te, seq_len)

    tr_loader = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # 6. Model, loss, optimizer
    model = SentinelCNN(
        n_features=n_features,
        seq_len=seq_len,
        dropout=cfg["dropout"],
    ).to(device)
    criterion = SentinelLoss(
        w_regime=cfg["w_regime"],
        w_anomaly=cfg["w_anomaly"],
        w_forecast=cfg["w_forecast"],
        class_weights=class_weights,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    # 7. Training loop with early stopping
    best_val_loss = float("inf")
    patience_ctr = 0
    history = []

    print(f"\nTraining for up to {cfg['epochs']} epochs...")
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, tr_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_epoch(model, va_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "tr_loss": tr_loss, "va_loss": va_loss,
                         "tr_acc": tr_acc, "va_acc": va_acc})

        print(f"  Epoch {epoch:3d} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.3f} | "
              f"{elapsed:.1f}s")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_ctr = 0
            torch.save(model.state_dict(),
                       os.path.join(cfg["artifacts_dir"], "sentinel_best.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    # 8. Train LightGBM baseline
    print("\nTraining LightGBM baseline...")
    lgbm_model = train_lgbm(X_tr, r_tr, X_va, r_va, pipeline.feature_names_, artifacts_dir=cfg["artifacts_dir"])
    
    print("\nEvaluating LightGBM baseline...")
    lgbm_metrics = evaluate_lgbm(lgbm_model, X_te, r_te, artifacts_dir=cfg["artifacts_dir"])

    # 9. Test evaluation (CNN)
    model.load_state_dict(
        torch.load(os.path.join(cfg["artifacts_dir"], "sentinel_best.pt"), map_location=device)
    )
    te_loss, te_acc = eval_epoch(model, te_loader, criterion, device)
    print(f"\nTest  | loss={te_loss:.4f}  acc={te_acc:.3f}")
    
    # Compute comprehensive metrics
    print("\nComputing comprehensive metrics...")
    comprehensive_metrics = compute_comprehensive_metrics(model, te_loader, device)
    
    # Print metrics to console
    print("\n" + "="*60)
    print("COMPREHENSIVE TEST METRICS")
    print("="*60)
    
    print("\n--- Regime Classification ---")
    print(f"  Macro F1: {comprehensive_metrics['regime_classification']['macro_f1']:.4f}")
    print(f"  Per-class Precision: {[f'{p:.4f}' for p in comprehensive_metrics['regime_classification']['per_class_precision']]}")
    print(f"  Per-class Recall:    {[f'{r:.4f}' for r in comprehensive_metrics['regime_classification']['per_class_recall']]}")
    print(f"  Confusion Matrix:")
    for row in comprehensive_metrics['regime_classification']['confusion_matrix']:
        print(f"    {row}")
    
    print("\n--- Anomaly Detection ---")
    print(f"  AUROC: {comprehensive_metrics['anomaly_detection']['auroc']:.4f}")
    print(f"  AUPRC: {comprehensive_metrics['anomaly_detection']['auprc']:.4f}")
    print(f"  Precision @ 95% Recall: {comprehensive_metrics['anomaly_detection']['precision_at_95_recall']:.4f}")
    
    print("\n--- Forecasting ---")
    print(f"  Per-sensor RMSE: {[f'{rmse:.4f}' for rmse in comprehensive_metrics['forecasting']['per_sensor_rmse']]}")
    print(f"  Per-sensor MAE:  {[f'{mae:.4f}' for mae in comprehensive_metrics['forecasting']['per_sensor_mae']]}")
    print("="*60 + "\n")
    
    # Save comprehensive metrics to JSON
    metrics_path = os.path.join(cfg["artifacts_dir"], "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(comprehensive_metrics, f, indent=2)
    print(f"Comprehensive metrics saved to {metrics_path}")
    
    # Print comparison table: CNN vs LightGBM
    print("\n" + "="*60)
    print("MODEL COMPARISON: CNN vs LightGBM")
    print("="*60)
    
    # Extract metrics for comparison
    cnn_acc = te_acc
    cnn_macro_f1 = comprehensive_metrics['regime_classification']['macro_f1']
    cnn_auroc = comprehensive_metrics['anomaly_detection']['auroc']
    cnn_auprc = comprehensive_metrics['anomaly_detection']['auprc']
    
    # LightGBM metrics from classification report
    lgbm_acc = lgbm_metrics['classification_report']['accuracy']
    lgbm_macro_f1 = lgbm_metrics['classification_report']['macro avg']['f1-score']
    lgbm_precision = lgbm_metrics['classification_report']['macro avg']['precision']
    lgbm_recall = lgbm_metrics['classification_report']['macro avg']['recall']
    
    print(f"\n{'Metric':<25} {'CNN':<12} {'LightGBM':<12}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {cnn_acc:<12.4f} {lgbm_acc:<12.4f}")
    print(f"{'Macro F1':<25} {cnn_macro_f1:<12.4f} {lgbm_macro_f1:<12.4f}")
    print(f"{'Macro Precision':<25} {'-':<12} {lgbm_precision:<12.4f}")
    print(f"{'Macro Recall':<25} {'-':<12} {lgbm_recall:<12.4f}")
    print(f"{'AUROC (Anomaly)':<25} {cnn_auroc:<12.4f} {'-':<12}")
    print(f"{'AUPRC (Anomaly)':<25} {cnn_auprc:<12.4f} {'-':<12}")
    print("="*60 + "\n")
    
    print("Note: CNN performs multi-task learning (regime + anomaly + forecast)")
    print("      LightGBM baseline only performs regime classification")
    print()

    # Run conformal calibration
    print("\nRunning conformal calibration...")
    run_calibration(model, te_loader, device, coverage=0.95, artifacts_dir=cfg["artifacts_dir"])
    print(f"Calibration thresholds saved to {cfg['artifacts_dir']}/thresholds.json")

    # 10. Save training history and model config
    run_meta = {
        "n_features": n_features,
        "seq_len": seq_len,
        "te_loss": te_loss,
        "te_acc": te_acc,
        "best_val_loss": best_val_loss,
        "config": cfg,
    }
    with open(os.path.join(cfg["artifacts_dir"], "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"\nArtifacts saved to {cfg['artifacts_dir']}/")
    return model, pipeline, run_meta


def load_config(path: str) -> dict:
    with open(path) as f:
        return {**DEFAULT_CFG, **yaml.safe_load(f)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--sim_hours",  type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CFG.copy()
    if os.path.exists(args.config):
        cfg.update(load_config(args.config))
    for key in ["epochs", "lr", "sim_hours", "batch_size"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    main(cfg)
