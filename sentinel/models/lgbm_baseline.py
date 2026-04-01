"""
LightGBM baseline for regime classification.

Uses tabular features (rolling stats, slopes, FFT) aggregated from the
same feature pipeline as the CNN.  Serves two purposes:

1. Comparison: shows when gradient boosting beats sequence models (and when it doesn't)
2. Explainability: SHAP values identify the most predictive sensor features

Usage
-----
    python -m sentinel.models.lgbm_baseline
"""

import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import shap
from sklearn.metrics import classification_report, confusion_matrix

from sentinel.models.cnn import REGIME_NAMES


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    feature_names: list[str],
    artifacts_dir: str = "artifacts",
) -> lgb.Booster:
    """
    Train a LightGBM classifier for regime classification.
    Each row in X is a SINGLE timestep (not a window) —
    this is the key architectural trade-off vs the CNN.
    """

    # Class weights for imbalanced data
    classes, counts = np.unique(y_train, return_counts=True)
    weights = {int(c): float(counts.sum() / (len(classes) * cnt))
               for c, cnt in zip(classes, counts)}
    sample_weights = np.array([weights[int(y)] for y in y_train])

    dtrain = lgb.Dataset(
        X_train, label=y_train,
        weight=sample_weights,
        feature_name=feature_names,
    )
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        "objective":        "multiclass",
        "num_class":        len(REGIME_NAMES),
        "metric":           "multi_logloss",
        "learning_rate":    0.05,
        "num_leaves":       63,
        "max_depth":        -1,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":          -1,
        "seed":             42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        lgb.log_evaluation(period=20),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    path = os.path.join(artifacts_dir, "lgbm_baseline.pkl")
    with open(path, "wb") as f:
        pickle.dump(booster, f)
    print(f"LightGBM model saved → {path}")
    return booster


def evaluate_lgbm(
    booster: lgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    artifacts_dir: str = "artifacts",
) -> dict:
    proba = booster.predict(X_test)
    preds = proba.argmax(axis=1)

    # Get unique labels present in test set (filter out phantom classes)
    unique_labels = np.unique(np.concatenate([y_test, preds]))
    target_names_filtered = [REGIME_NAMES[i] for i in unique_labels]

    report = classification_report(
        y_test, preds,
        labels=unique_labels,
        target_names=target_names_filtered,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=unique_labels)

    print("\nLightGBM Classification Report:")
    print(classification_report(y_test, preds, labels=unique_labels, target_names=target_names_filtered, zero_division=0))

    result = {"classification_report": report, "confusion_matrix": cm.tolist()}
    with open(os.path.join(artifacts_dir, "lgbm_eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def compute_shap(
    booster: lgb.Booster,
    X_sample: np.ndarray,
    feature_names: list[str],
    artifacts_dir: str = "artifacts",
    max_samples: int = 500,
) -> np.ndarray:
    """
    Compute SHAP values and save summary to disk.
    Returns shap_values array of shape (n_samples, n_features, n_classes).
    """
    X_sub = X_sample[:max_samples]
    explainer   = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sub)   # list of arrays per class

    # Mean absolute SHAP per feature (averaged over classes)
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    importance = sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda x: x[1], reverse=True,
    )[:20]

    print("\nTop 10 features by mean |SHAP|:")
    for name, val in importance[:10]:
        print(f"  {name:<35} {val:.5f}")

    shap_summary = {
        "top_features": [{"feature": n, "mean_abs_shap": v} for n, v in importance]
    }
    with open(os.path.join(artifacts_dir, "shap_summary.json"), "w") as f:
        json.dump(shap_summary, f, indent=2)

    return shap_values


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from sentinel.datagen.pump_sim import PumpConfig, simulate
    from sentinel.features.pipeline import FeaturePipeline

    cfg = PumpConfig(duration_hours=50, seed=7)
    df  = simulate(cfg)

    pipeline = FeaturePipeline()
    X, labels = pipeline.fit_transform(df, df["regime_label"])
    feat_names = pipeline.feature_names_

    n = len(X)
    i1, i2 = int(n * 0.7), int(n * 0.85)
    X_tr, y_tr = X[:i1],    labels[:i1]
    X_va, y_va = X[i1:i2],  labels[i1:i2]
    X_te, y_te = X[i2:],    labels[i2:]

    os.makedirs("artifacts", exist_ok=True)
    booster = train_lgbm(X_tr, y_tr, X_va, y_va, feat_names)
    evaluate_lgbm(booster, X_te, y_te)
    compute_shap(booster, X_te, feat_names)
