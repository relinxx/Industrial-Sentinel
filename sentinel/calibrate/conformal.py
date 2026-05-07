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
    Compute per-regime conformal anomaly thresholds.

    For each regime, we treat the anomaly scores from that regime's
    calibration samples as the null distribution and compute the
    coverage-guaranteed threshold.

    Parameters
    ----------
    anomaly_scores : (N,) float32 sigmoid outputs from the model
    regime_labels  : (N,) int regime predictions from the model
    coverage       : desired coverage (default 0.95)

    Returns
    -------
    dict with keys:
        "global"              : float threshold over all samples
        "per_regime"          : dict regime_name → threshold
        "coverage"            : float requested coverage
        "n_calibration_total" : int total samples used
    """
    thresholds = {}
    for i, name in enumerate(REGIME_NAMES):
        mask = regime_labels == i
        regime_scores = anomaly_scores[mask]
        if len(regime_scores) < 10:
            thresholds[name] = conformal_threshold(anomaly_scores, coverage)
        else:
            thresholds[name] = conformal_threshold(regime_scores, coverage)

    global_q = conformal_threshold(anomaly_scores, coverage)

    return {
        "global":               global_q,
        "per_regime":           thresholds,
        "coverage":             coverage,
        "n_calibration_total":  len(anomaly_scores),
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
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    all_scores  = []
    all_regimes = []

    with torch.no_grad():
        for x, reg, anom, fcast in loader:
            x = x.to(device)
            logits, anom_score, _ = model(x)
            all_scores.append(anom_score.squeeze(-1).cpu().numpy())
            all_regimes.append(logits.argmax(dim=-1).cpu().numpy())

    scores  = np.concatenate(all_scores)
    regimes = np.concatenate(all_regimes)

    result = conformal_thresholds(scores, regimes, coverage)

    print(f"\nConformal calibration (coverage={coverage}):")
    print(f"  Global threshold  : {result['global']:.4f}")
    for name, thr in result["per_regime"].items():
        count = result["regime_counts"][name]
        print(f"  {name:<18}: {thr:.4f}  (n={count})")

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
