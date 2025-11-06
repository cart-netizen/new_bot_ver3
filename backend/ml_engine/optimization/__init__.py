"""
ML Model Optimization Module

Components:
- ONNX Optimizer: PyTorch → ONNX export and optimization
- Quantization: FP32 → INT8
- Benchmarking tools
"""

from .onnx_optimizer import ONNXOptimizer, get_onnx_optimizer

__all__ = [
    "ONNXOptimizer",
    "get_onnx_optimizer"
]
