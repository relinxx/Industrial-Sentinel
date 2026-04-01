"""
Test suite for Industrial-Sentinel ML pipeline.

Tests validate:
- Feature engineering correctness (no data leakage, correct window sizes)
- Model forward pass behavior (output shapes, value ranges)
- ONNX export parity with PyTorch
"""
