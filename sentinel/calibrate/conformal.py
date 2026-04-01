"""
Conformal prediction calibration for anomaly thresholds.

Computes coverage-guaranteed anomaly thresholds per operating regime
using split conformal prediction on a held-out calibration set.

Reference: Angelopoulos & Bates (2022) "A Gentle Introduction to Conformal Prediction"

Usage
-----
    from sentinel.calibrate.conformal import conformal_thresholds
    thresholds = conformal_thresholds(anomaly_scores, regime_labels, coverage=0.95)
"""

import json
import os
from typing import Optional

import numpy as np

from sentinel.models.cnn import REGIME_NAMES


def conformal_threshold(
    scores: np.ndarray,
    coverage: float = 0.95,
) -> float:
    """
    Compute the conformal quantile threshold for a 1-D score array.

    Returns the smallest threshold q such that at least `coverage` fraction
    of calibration samples satisfy score <= q.

    Parameters
    ----------
    scores   : anomaly scores for non-anomalous (healthy) calibration samples
    coverage : desired coverage level (e.g. 0.95 → 95% of healthy samples below threshold)
    """
    n = len(scores)
    if n == 0:
        return 0.5
    q_level = np.ceil((n + 1) * coverage) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level))


def conformal_thresholds(
    anomaly_scores: np.ndarray,
    regime_labels:  np.ndarray,
    coverage: float = 0.95,
) -> dict:
    """
    Compute conformal anomaly threshold using HEALTHY samples only.

    The null distribution for anomaly detection is the set of healthy (regime 0)
    samples. The threshold answers: "at what anomaly score value do 95% of 
    healthy samples fall below?"

    Parameters
    ----------
    anomaly_scores : (N,) float32 sigmoid outputs from the model
    regime_labels  : (N,) int TRUE regime labels (not predictions)
    coverage       : desired coverage (default 0.95)

    Returns
    -------
    dict with keys:
        "threshold"           : float global threshold based on healthy samples
        "coverage"            : float requested coverage
        "n_calibration_total" : int total samples used
        "n_healthy"           : int number of healthy samples used
    """
    # Null distribution = healthy samples only (regime 0)
    healthy_mask = regime_labels == 0
    healthy_scores = anomaly_scores[healthy_mask]
    
    if len(healthy_scores) < 10:
        raise ValueError(
            f"Insufficient healthy samples for calibration: {len(healthy_scores)} < 10. "
            "Need more regime 0 (healthy) samples in calibration set."
        )
    
    threshold = conformal_threshold(healthy_scores, coverage)

    return {
        "threshold":            threshold,
        "coverage":             coverage,
        "n_calibration_total":  len(anomaly_scores),
        "n_healthy":            int(healthy_mask.sum()),
        "regime_counts":        {
            name: int((regime_labels == i).sum())
            for i, name in enumerate(REGIME_NAMES)
        },
    }


def run_calibration(
    model,
    loader,
    device,
    coverage: float = 0.95,
    artifacts_dir: str = "artifacts",
) -> dict:
    """
    Run calibration on a DataLoader and save threshold config to disk.
    
    Uses TRUE regime labels from the loader to identify healthy samples,
    then computes a single global threshold based on the null distribution
    (healthy samples only).
    """
    import torch

    model.eval()
    all_scores  = []
    all_true_regimes = []

    with torch.no_grad():
        for x, reg, anom, fcast in loader:
            x = x.to(device)
            _, anom_score, _ = model(x)
            all_scores.append(anom_score.squeeze(-1).cpu().numpy())
            # Use TRUE labels from loader, not predictions
            all_true_regimes.append(reg.numpy())

    scores  = np.concatenate(all_scores)
    regimes = np.concatenate(all_true_regimes)

    result = conformal_thresholds(scores, regimes, coverage)

    print(f"\nConformal calibration (coverage={coverage}):")
    print(f"  Threshold (healthy): {result['threshold']:.6f}")
    print(f"  Healthy samples    : {result['n_healthy']}")
    print(f"  Total samples      : {result['n_calibration_total']}")
    print(f"\nRegime distribution:")
    for name, count in zip(REGIME_NAMES, [result['regime_counts'][n] for n in REGIME_NAMES]):
        print(f"  {name:<18}: {count}")

    path = os.path.join(artifacts_dir, "thresholds.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nThresholds saved → {path}")

    return result


if __name__ == "__main__":
    # Demo with synthetic scores
    rng = np.random.default_rng(42)
    scores  = rng.beta(2, 5, 1000).astype(np.float32)
    regimes = rng.integers(0, 4, 1000)
    result  = conformal_thresholds(scores, regimes, coverage=0.95)
    print(json.dumps(result, indent=2))
