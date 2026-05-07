"""
ONNX export for IndustrialSentinel.

Exports the trained SentinelCNN to ONNX opset 17,
validates numerical parity with PyTorch,
and writes inference_contract.json.

Usage
-----
    python -m sentinel.export.onnx_export
    python -m sentinel.export.onnx_export --model artifacts/sentinel_best.pt
"""

import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from sentinel.models.cnn import SentinelCNN, FORECAST_STEPS, N_RAW_SENSORS, REGIME_NAMES


OPSET = 17


def export_to_onnx(
    model: SentinelCNN,
    onnx_path: str,
    seq_len: int,
    n_features: int,
    batch_size: int = 1,
) -> None:
    model.eval()
    dummy = torch.randn(batch_size, seq_len, n_features)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=OPSET,
        input_names=["sensor_window"],
        output_names=["regime_logits", "anomaly_score", "forecast"],
        dynamic_axes={
            "sensor_window":  {0: "batch"},
            "regime_logits":  {0: "batch"},
            "anomaly_score":  {0: "batch"},
            "forecast":       {0: "batch"},
        },
        export_params=True,
    )
    print(f"  ONNX model written → {onnx_path}")


def validate_onnx(
    model: SentinelCNN,
    onnx_path: str,
    seq_len: int,
    n_features: int,
    n_samples: int = 10,
    atol: float = 1e-4,
) -> bool:
    """Check that ONNX Runtime outputs match PyTorch outputs within atol."""
    model.eval()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    all_match = True
    rng = np.random.default_rng(0)
    for i in range(n_samples):
        x_np = rng.standard_normal((1, seq_len, n_features)).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        with torch.no_grad():
            pt_logits, pt_anom, pt_fcast = model(x_pt)
            pt_probs = F.softmax(pt_logits, dim=-1).numpy()
            pt_anom_np  = pt_anom.numpy()
            pt_fcast_np = pt_fcast.numpy()

        ort_out = sess.run(
            None, {"sensor_window": x_np}
        )
        ort_logits, ort_anom, ort_fcast = ort_out
        ort_probs = np.exp(ort_logits) / np.exp(ort_logits).sum(-1, keepdims=True)

        # ONNX exports raw logits — apply softmax here for comparison
        ort_probs = np.exp(ort_logits - ort_logits.max(-1, keepdims=True))
        ort_probs = ort_probs / ort_probs.sum(-1, keepdims=True)

        close = (
            np.allclose(pt_probs, ort_probs, atol=atol) and
            np.allclose(pt_anom_np, ort_anom, atol=atol) and
            np.allclose(pt_fcast_np, ort_fcast, atol=atol)
        )
        if not close:
            print(f"  [WARN] Sample {i}: parity check FAILED")
            all_match = False

    if all_match:
        print(f"  Parity check PASSED on {n_samples} random inputs (atol={atol})")
    return all_match


def write_inference_contract(
    onnx_path: str,
    scaler_path: str,
    feature_spec_path: str,
    seq_len: int,
    n_features: int,
    thresholds: dict | None = None,
    output_path: str = "artifacts/inference_contract.json",
) -> None:
    """
    Write a machine-readable inference contract for .NET ONNX Runtime integration.
    The C# team can import this JSON and use it to configure their inference pipeline.
    """
    scaler_cfg = {}
    if os.path.exists(scaler_path):
        with open(scaler_path) as f:
            scaler_cfg = json.load(f)

    contract = {
        "version": "1.0",
        "model_file": os.path.basename(onnx_path),
        "onnx_opset": OPSET,
        "framework": "PyTorch → ONNX Runtime",

        "input": {
            "name":        "sensor_window",
            "shape":       ["batch", seq_len, n_features],
            "dtype":       "float32",
            "description": (
                f"Sliding window of {seq_len} timesteps × {n_features} engineered features. "
                "Features must be scaled with the provided scaler_config before inference."
            ),
            "seq_len":    seq_len,
            "n_features": n_features,
        },

        "outputs": {
            "regime_logits": {
                "name":        "regime_logits",
                "shape":       ["batch", len(REGIME_NAMES)],
                "dtype":       "float32",
                "description": "Raw logits — apply softmax to get class probabilities.",
                "labels":      REGIME_NAMES,
                "label_indices": {r: i for i, r in enumerate(REGIME_NAMES)},
                "post_process": "softmax",
            },
            "anomaly_score": {
                "name":        "anomaly_score",
                "shape":       ["batch", 1],
                "dtype":       "float32",
                "range":       [0.0, 1.0],
                "description": "Anomaly probability (sigmoid applied in model). "
                               "Higher = more anomalous.",
                "recommended_threshold": (thresholds or {}).get("anomaly_global", 0.5),
                "regime_thresholds": (thresholds or {}).get("anomaly_per_regime", {}),
            },
            "forecast": {
                "name":        "forecast",
                "shape":       ["batch", FORECAST_STEPS, N_RAW_SENSORS],
                "dtype":       "float32",
                "description": f"Next {FORECAST_STEPS} timestep forecast of raw sensor values.",
                "forecast_steps": FORECAST_STEPS,
                "sensor_order":   [
                    "discharge_pressure [bar]",
                    "suction_pressure [bar]",
                    "flow_rate [m3/h]",
                    "motor_current [A]",
                    "vibration_rms [mm/s]",
                    "bearing_temp [C]",
                ],
            },
        },

        "preprocessing": {
            "scaler_type":    scaler_cfg.get("type", "StandardScaler"),
            "scaler_config":  scaler_path,
            "feature_spec":   feature_spec_path,
            "warmup_steps":   max(30, 20, 64),
            "description":    (
                "1. Resample to 1 Hz. "
                "2. Impute missing values (forward-fill, then zero-fill). "
                "3. Compute engineered features per feature_spec.json. "
                "4. Apply StandardScaler: x_scaled = (x - mean) / std. "
                "5. Buffer seq_len=60 steps into a sliding window. "
                "6. Pass window to ONNX Runtime as float32 tensor."
            ),
        },

        "runtime": {
            "tested_provider":  "CPUExecutionProvider",
            "dotnet_package":   "Microsoft.ML.OnnxRuntime",
            "min_dotnet_version": "net6.0",
            "sample_csharp": (
                "var session = new InferenceSession(\"sentinel.onnx\");\n"
                "var inputTensor = new DenseTensor<float>(data, new[] { 1, 60, " + str(n_features) + " });\n"
                "var inputs = new List<NamedOnnxValue> {\n"
                "    NamedOnnxValue.CreateFromTensor(\"sensor_window\", inputTensor)\n"
                "};\n"
                "using var results = session.Run(inputs);\n"
                "var regimeLogits = results.First(r => r.Name == \"regime_logits\").AsTensor<float>();\n"
                "var anomalyScore = results.First(r => r.Name == \"anomaly_score\").AsTensor<float>();"
            ),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(contract, f, indent=2)
    print(f"  Inference contract written → {output_path}")


def run_export(
    model_pt_path: str = "artifacts/sentinel_best.pt",
    artifacts_dir: str = "artifacts",
):
    meta_path = os.path.join(artifacts_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"run_meta.json not found in {artifacts_dir}. Run train.py first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    n_features = meta["n_features"]
    seq_len    = meta["seq_len"]

    print(f"Loading model  : {model_pt_path}  (n_features={n_features}, seq_len={seq_len})")
    model = SentinelCNN(n_features=n_features, seq_len=seq_len)
    model.load_state_dict(torch.load(model_pt_path, map_location="cpu"))
    model.eval()

    onnx_path = os.path.join(artifacts_dir, "sentinel.onnx")

    print("Exporting to ONNX...")
    export_to_onnx(model, onnx_path, seq_len, n_features)

    print("Validating ONNX parity...")
    ok = validate_onnx(model, onnx_path, seq_len, n_features)
    if not ok:
        print("  [WARN] Parity check failed — review model for unsupported ops")

    print("Writing inference contract...")
    write_inference_contract(
        onnx_path=onnx_path,
        scaler_path=os.path.join(artifacts_dir, "scaler_config.json"),
        feature_spec_path=os.path.join(artifacts_dir, "feature_spec.json"),
        seq_len=seq_len,
        n_features=n_features,
        thresholds={"anomaly_global": 0.5},
        output_path=os.path.join(artifacts_dir, "inference_contract.json"),
    )

    print("\nDone. Artifacts:")
    for f in ["sentinel.onnx", "scaler_config.json", "feature_spec.json",
              "inference_contract.json", "run_meta.json"]:
        p = os.path.join(artifacts_dir, f)
        if os.path.exists(p):
            size_kb = os.path.getsize(p) / 1024
            print(f"  {f:<35} {size_kb:6.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/sentinel_best.pt")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    run_export(args.model, args.artifacts)
