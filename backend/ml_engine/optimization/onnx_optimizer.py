"""
ONNX Optimizer - экспорт PyTorch моделей в ONNX и оптимизация

Функции:
- Экспорт PyTorch → ONNX
- Quantization (FP32 → INT8)
- Graph optimization
- Benchmarking
"""

import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX/ONNXRuntime not installed. Install with: pip install onnx onnxruntime")

from backend.core.logger import get_logger

logger = get_logger(__name__)


class ONNXOptimizer:
    """
    ONNX Optimizer для оптимизации ML моделей

    Цели:
    - Latency: < 3ms (vs ~5ms для PyTorch)
    - Memory: -30% usage
    - Throughput: +50%
    """

    def __init__(self):
        if not ONNX_AVAILABLE:
            logger.warning("ONNX not available, optimizer will not work")
        else:
            logger.info("ONNX Optimizer initialized")

    async def export_to_onnx(
        self,
        model: nn.Module,
        model_path: Path,
        output_path: Path,
        input_shape: Tuple[int, ...],
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Экспорт PyTorch модели в ONNX

        Args:
            model: PyTorch модель
            model_path: Путь к .pt файлу (для загрузки весов)
            output_path: Путь для сохранения .onnx
            input_shape: Форма входа (batch_size, timesteps, features)
            opset_version: Версия ONNX opset
            dynamic_axes: Динамические оси (для batch размера)

        Returns:
            True если успешно
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available")
            return False

        try:
            # Загрузить веса модели
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded model weights from {model_path}")

            # Перевести в eval mode
            model.eval()

            # Создать dummy input
            dummy_input = torch.randn(*input_shape)

            # Dynamic axes для batch size
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }

            # Экспорт
            logger.info(f"Exporting to ONNX: {output_path}")
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )

            # Проверка валидности
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"ONNX export successful ({file_size_mb:.2f} MB)")

            return True

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False

    async def quantize_model(
        self,
        onnx_path: Path,
        output_path: Path,
        quantization_type: str = "dynamic"
    ) -> bool:
        """
        Quantize ONNX модель (FP32 → INT8)

        Args:
            onnx_path: Путь к ONNX модели
            output_path: Путь для quantized модели
            quantization_type: "dynamic" или "static"

        Returns:
            True если успешно
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available")
            return False

        if not onnx_path.exists():
            logger.error(f"ONNX model not found: {onnx_path}")
            return False

        try:
            logger.info(f"Quantizing {onnx_path} → {output_path}")

            # Dynamic quantization (не требует калибровочных данных)
            if quantization_type == "dynamic":
                quantize_dynamic(
                    str(onnx_path),
                    str(output_path),
                    weight_type=QuantType.QUInt8
                )

            original_size = onnx_path.stat().st_size / (1024 * 1024)
            quantized_size = output_path.stat().st_size / (1024 * 1024)
            reduction = (1 - quantized_size / original_size) * 100

            logger.info(
                f"Quantization successful: "
                f"{original_size:.2f} MB → {quantized_size:.2f} MB "
                f"({reduction:.1f}% reduction)"
            )

            return True

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False

    async def optimize_graph(self, onnx_path: Path, output_path: Path) -> bool:
        """
        Оптимизация графа ONNX модели

        Args:
            onnx_path: Путь к ONNX модели
            output_path: Путь для оптимизированной модели

        Returns:
            True если успешно
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available")
            return False

        try:
            # Загрузить модель
            model = onnx.load(str(onnx_path))

            # Применить оптимизации
            # - Constant folding
            # - Dead code elimination
            # - Common subexpression elimination
            from onnx import optimizer

            passes = [
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_transpose',
                'fuse_consecutive_transposes',
                'fuse_transpose_into_gemm',
            ]

            optimized_model = optimizer.optimize(model, passes)

            # Сохранить
            onnx.save(optimized_model, str(output_path))

            logger.info(f"Graph optimization successful: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Graph optimization failed: {e}")
            return False

    async def benchmark(
        self,
        onnx_path: Path,
        input_shape: Tuple[int, ...],
        num_iterations: int = 1000,
        warmup_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark ONNX модели

        Args:
            onnx_path: Путь к ONNX модели
            input_shape: Форма входа
            num_iterations: Количество итераций
            warmup_iterations: Количество warmup итераций

        Returns:
            Словарь с метриками:
            - latency_ms: среднее время inference
            - throughput: predictions/sec
            - p50_ms, p95_ms, p99_ms: перцентили latency
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available")
            return {}

        if not onnx_path.exists():
            logger.error(f"ONNX model not found: {onnx_path}")
            return {}

        try:
            # Создать session
            session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )

            # Получить input name
            input_name = session.get_inputs()[0].name

            # Создать dummy data
            dummy_input = np.random.randn(*input_shape).astype(np.float32)

            # Warmup
            logger.info(f"Warmup: {warmup_iterations} iterations")
            for _ in range(warmup_iterations):
                session.run(None, {input_name: dummy_input})

            # Benchmark
            logger.info(f"Benchmarking: {num_iterations} iterations")
            latencies = []

            for _ in range(num_iterations):
                start_time = time.perf_counter()
                session.run(None, {input_name: dummy_input})
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # ms

            # Вычислить метрики
            latencies_np = np.array(latencies)
            metrics = {
                "latency_ms": float(np.mean(latencies_np)),
                "latency_std_ms": float(np.std(latencies_np)),
                "p50_ms": float(np.percentile(latencies_np, 50)),
                "p95_ms": float(np.percentile(latencies_np, 95)),
                "p99_ms": float(np.percentile(latencies_np, 99)),
                "min_ms": float(np.min(latencies_np)),
                "max_ms": float(np.max(latencies_np)),
                "throughput": 1000 / np.mean(latencies_np)  # predictions/sec
            }

            logger.info(
                f"Benchmark results: "
                f"avg={metrics['latency_ms']:.2f}ms, "
                f"p95={metrics['p95_ms']:.2f}ms, "
                f"throughput={metrics['throughput']:.0f}/sec"
            )

            return metrics

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {}

    async def compare_pytorch_onnx(
        self,
        pytorch_model: nn.Module,
        pytorch_weights_path: Path,
        onnx_path: Path,
        input_shape: Tuple[int, ...],
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Сравнить PyTorch и ONNX версии модели

        Args:
            pytorch_model: PyTorch модель
            pytorch_weights_path: Путь к весам PyTorch
            onnx_path: Путь к ONNX модели
            input_shape: Форма входа
            num_samples: Количество samples для сравнения

        Returns:
            Словарь с результатами сравнения
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available")
            return {}

        try:
            # Загрузить PyTorch модель
            checkpoint = torch.load(pytorch_weights_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                pytorch_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                pytorch_model.load_state_dict(checkpoint)
            pytorch_model.eval()

            # Загрузить ONNX модель
            ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            input_name = ort_session.get_inputs()[0].name

            # Сравнить выходы
            differences = []
            pytorch_times = []
            onnx_times = []

            with torch.no_grad():
                for _ in range(num_samples):
                    # Создать random input
                    x = torch.randn(*input_shape)
                    x_np = x.numpy().astype(np.float32)

                    # PyTorch inference
                    start = time.perf_counter()
                    pytorch_out = pytorch_model(x)
                    pytorch_times.append((time.perf_counter() - start) * 1000)

                    # ONNX inference
                    start = time.perf_counter()
                    onnx_out = ort_session.run(None, {input_name: x_np})[0]
                    onnx_times.append((time.perf_counter() - start) * 1000)

                    # Вычислить разницу
                    pytorch_np = pytorch_out.numpy()
                    diff = np.abs(pytorch_np - onnx_out)
                    differences.append(np.mean(diff))

            results = {
                "numerical_difference": {
                    "mean": float(np.mean(differences)),
                    "max": float(np.max(differences)),
                    "std": float(np.std(differences))
                },
                "pytorch_latency_ms": {
                    "mean": float(np.mean(pytorch_times)),
                    "p95": float(np.percentile(pytorch_times, 95))
                },
                "onnx_latency_ms": {
                    "mean": float(np.mean(onnx_times)),
                    "p95": float(np.percentile(onnx_times, 95))
                },
                "speedup": float(np.mean(pytorch_times) / np.mean(onnx_times)),
                "outputs_match": np.mean(differences) < 1e-5
            }

            logger.info(
                f"Comparison: "
                f"PyTorch={results['pytorch_latency_ms']['mean']:.2f}ms, "
                f"ONNX={results['onnx_latency_ms']['mean']:.2f}ms, "
                f"Speedup={results['speedup']:.2f}x"
            )

            return results

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {}

    async def export_and_optimize(
        self,
        model: nn.Module,
        model_path: Path,
        output_dir: Path,
        input_shape: Tuple[int, ...],
        quantize: bool = True,
        benchmark_iterations: int = 1000
    ) -> Dict[str, Any]:
        """
        Полный pipeline: экспорт → оптимизация → quantization → benchmark

        Args:
            model: PyTorch модель
            model_path: Путь к весам модели
            output_dir: Директория для сохранения результатов
            input_shape: Форма входа
            quantize: Применить quantization
            benchmark_iterations: Количество итераций для benchmark

        Returns:
            Словарь с результатами
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "export_success": False,
            "quantize_success": False,
            "benchmarks": {}
        }

        # 1. Export to ONNX
        onnx_path = output_dir / "model.onnx"
        export_success = await self.export_to_onnx(
            model, model_path, onnx_path, input_shape
        )
        results["export_success"] = export_success

        if not export_success:
            return results

        # 2. Benchmark original ONNX
        results["benchmarks"]["onnx_fp32"] = await self.benchmark(
            onnx_path, input_shape, benchmark_iterations
        )

        # 3. Quantization (optional)
        if quantize:
            quantized_path = output_dir / "model_quantized.onnx"
            quantize_success = await self.quantize_model(
                onnx_path, quantized_path
            )
            results["quantize_success"] = quantize_success

            if quantize_success:
                # Benchmark quantized
                results["benchmarks"]["onnx_int8"] = await self.benchmark(
                    quantized_path, input_shape, benchmark_iterations
                )

        # 4. Compare PyTorch vs ONNX
        comparison = await self.compare_pytorch_onnx(
            model, model_path, onnx_path, input_shape
        )
        results["comparison"] = comparison

        logger.info("Export and optimization pipeline completed")
        return results


# Singleton instance
_optimizer_instance: Optional[ONNXOptimizer] = None


def get_onnx_optimizer() -> ONNXOptimizer:
    """Получить singleton instance ONNX Optimizer"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = ONNXOptimizer()
    return _optimizer_instance
