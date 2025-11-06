"""
Integration tests для ML Model Serving Infrastructure

Tests:
- Model Registry
- ONNX Optimizer
- A/B Testing
- Model Server (endpoints)
"""

import pytest
import asyncio
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from backend.ml_engine.inference.model_registry import (
    ModelRegistry,
    ModelStage
)
from backend.ml_engine.inference.ab_testing import (
    ABTestManager,
    ModelVariant,
    PredictionOutcome
)
from backend.ml_engine.optimization.onnx_optimizer import ONNXOptimizer
from backend.ml_engine.models.hybrid_cnn_lstm import HybridCNNLSTM


@pytest.fixture
async def model_registry():
    """Fixture для Model Registry"""
    registry = ModelRegistry(registry_dir="test_models")
    yield registry
    # Cleanup
    import shutil
    if Path("test_models").exists():
        shutil.rmtree("test_models")


@pytest.fixture
def sample_model():
    """Fixture для sample модели"""
    model = HybridCNNLSTM(
        input_size=110,
        lstm_hidden_size=128,
        lstm_layers=1,
        cnn_channels=[32, 64],
        kernel_sizes=[3, 5],
        dropout=0.2
    )
    return model


@pytest.fixture
async def ab_test_manager():
    """Fixture для A/B Test Manager"""
    manager = ABTestManager()
    yield manager


class TestModelRegistry:
    """Tests для Model Registry"""

    @pytest.mark.asyncio
    async def test_register_model(self, model_registry, sample_model, tmp_path):
        """Test регистрации модели"""
        # Сохранить модель во временный файл
        model_path = tmp_path / "test_model.pt"
        torch.save(sample_model.state_dict(), model_path)

        # Регистрация
        model_info = await model_registry.register_model(
            name="test_model",
            version="1.0.0",
            model_path=model_path,
            model_type="HybridCNNLSTM",
            description="Test model",
            metrics={"accuracy": 0.85, "sharpe": 2.5},
            tags=["test", "hybrid"]
        )

        assert model_info is not None
        assert model_info.metadata.name == "test_model"
        assert model_info.metadata.version == "1.0.0"
        assert model_info.metadata.metrics["accuracy"] == 0.85
        assert model_info.model_exists()

    @pytest.mark.asyncio
    async def test_get_model(self, model_registry, sample_model, tmp_path):
        """Test получения модели"""
        # Регистрируем модель
        model_path = tmp_path / "test_model.pt"
        torch.save(sample_model.state_dict(), model_path)

        await model_registry.register_model(
            name="test_model",
            version="1.0.0",
            model_path=model_path,
            model_type="HybridCNNLSTM"
        )

        # Получаем модель
        model_info = await model_registry.get_model("test_model", "1.0.0")

        assert model_info is not None
        assert model_info.metadata.name == "test_model"

    @pytest.mark.asyncio
    async def test_set_model_stage(self, model_registry, sample_model, tmp_path):
        """Test установки стадии модели"""
        # Регистрируем модель
        model_path = tmp_path / "test_model.pt"
        torch.save(sample_model.state_dict(), model_path)

        await model_registry.register_model(
            name="test_model",
            version="1.0.0",
            model_path=model_path,
            model_type="HybridCNNLSTM"
        )

        # Устанавливаем staging
        success = await model_registry.set_model_stage(
            "test_model", "1.0.0", ModelStage.STAGING
        )
        assert success

        # Получаем staging модель
        model_info = await model_registry.get_staging_model("test_model")
        assert model_info is not None
        assert model_info.metadata.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_promote_to_production(self, model_registry, sample_model, tmp_path):
        """Test продвижения в production"""
        # Регистрируем модель
        model_path = tmp_path / "test_model.pt"
        torch.save(sample_model.state_dict(), model_path)

        await model_registry.register_model(
            name="test_model",
            version="1.0.0",
            model_path=model_path,
            model_type="HybridCNNLSTM"
        )

        # Продвигаем в production
        success = await model_registry.promote_to_production("test_model", "1.0.0")
        assert success

        # Получаем production модель
        model_info = await model_registry.get_production_model("test_model")
        assert model_info is not None
        assert model_info.metadata.stage == ModelStage.PRODUCTION

    @pytest.mark.asyncio
    async def test_list_models(self, model_registry, sample_model, tmp_path):
        """Test списка моделей"""
        # Регистрируем несколько версий
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            model_path = tmp_path / f"model_{version}.pt"
            torch.save(sample_model.state_dict(), model_path)

            await model_registry.register_model(
                name="test_model",
                version=version,
                model_path=model_path,
                model_type="HybridCNNLSTM"
            )

        # Список всех версий
        models = await model_registry.list_models("test_model")
        assert len(models) == 3

        versions = [m.metadata.version for m in models]
        assert "1.0.0" in versions
        assert "1.1.0" in versions
        assert "2.0.0" in versions

    @pytest.mark.asyncio
    async def test_update_metrics(self, model_registry, sample_model, tmp_path):
        """Test обновления метрик"""
        # Регистрируем модель
        model_path = tmp_path / "test_model.pt"
        torch.save(sample_model.state_dict(), model_path)

        await model_registry.register_model(
            name="test_model",
            version="1.0.0",
            model_path=model_path,
            model_type="HybridCNNLSTM",
            metrics={"accuracy": 0.80}
        )

        # Обновляем метрики
        success = await model_registry.update_metrics(
            "test_model",
            "1.0.0",
            {"accuracy": 0.85, "sharpe": 2.5}
        )
        assert success

        # Проверяем
        model_info = await model_registry.get_model("test_model", "1.0.0")
        assert model_info.metadata.metrics["accuracy"] == 0.85
        assert model_info.metadata.metrics["sharpe"] == 2.5


class TestABTesting:
    """Tests для A/B Testing"""

    @pytest.mark.asyncio
    async def test_create_experiment(self, ab_test_manager):
        """Test создания эксперимента"""
        success = await ab_test_manager.create_experiment(
            experiment_id="test_exp_1",
            control_model_name="model_v1",
            control_model_version="1.0.0",
            treatment_model_name="model_v2",
            treatment_model_version="1.1.0",
            control_traffic=0.8,
            treatment_traffic=0.2
        )

        assert success
        assert "test_exp_1" in ab_test_manager.experiments

    @pytest.mark.asyncio
    async def test_traffic_routing(self, ab_test_manager):
        """Test traffic routing"""
        await ab_test_manager.create_experiment(
            experiment_id="test_exp_2",
            control_model_name="model_v1",
            control_model_version="1.0.0",
            treatment_model_name="model_v2",
            treatment_model_version="1.1.0",
            control_traffic=0.9,
            treatment_traffic=0.1
        )

        # Симулируем 1000 requests
        control_count = 0
        treatment_count = 0

        for _ in range(1000):
            variant = await ab_test_manager.route_traffic("test_exp_2")
            if variant == ModelVariant.CONTROL:
                control_count += 1
            else:
                treatment_count += 1

        # Должно быть примерно 90/10 split
        assert 850 < control_count < 950
        assert 50 < treatment_count < 150

    @pytest.mark.asyncio
    async def test_record_prediction(self, ab_test_manager):
        """Test записи predictions"""
        await ab_test_manager.create_experiment(
            experiment_id="test_exp_3",
            control_model_name="model_v1",
            control_model_version="1.0.0",
            treatment_model_name="model_v2",
            treatment_model_version="1.1.0"
        )

        # Записываем prediction
        outcome = PredictionOutcome(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            prediction={"direction": "BUY", "confidence": 0.85},
            actual_outcome=0.02,  # +2% return
            latency_ms=3.5
        )

        await ab_test_manager.record_prediction(
            "test_exp_3",
            ModelVariant.CONTROL,
            outcome
        )

        # Проверяем metrics
        metrics = ab_test_manager.metrics["test_exp_3"][ModelVariant.CONTROL]
        assert metrics.total_predictions == 1
        assert len(metrics.latencies) == 1
        assert metrics.latencies[0] == 3.5

    @pytest.mark.asyncio
    async def test_analyze_experiment(self, ab_test_manager):
        """Test анализа эксперимента"""
        # Создаем эксперимент
        await ab_test_manager.create_experiment(
            experiment_id="test_exp_4",
            control_model_name="model_v1",
            control_model_version="1.0.0",
            treatment_model_name="model_v2",
            treatment_model_version="1.1.0"
        )

        # Симулируем predictions для control
        for i in range(100):
            outcome = PredictionOutcome(
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                prediction={"direction": "BUY"},
                actual_outcome=0.01 if i % 2 == 0 else -0.01,
                latency_ms=4.0
            )
            await ab_test_manager.record_prediction(
                "test_exp_4",
                ModelVariant.CONTROL,
                outcome
            )

        # Симулируем predictions для treatment (лучше accuracy)
        for i in range(100):
            outcome = PredictionOutcome(
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                prediction={"direction": "BUY"},
                actual_outcome=0.01 if i % 3 != 0 else -0.01,  # ~67% win rate
                latency_ms=3.5
            )
            await ab_test_manager.record_prediction(
                "test_exp_4",
                ModelVariant.TREATMENT,
                outcome
            )

        # Анализ
        analysis = await ab_test_manager.analyze_experiment("test_exp_4")

        assert analysis is not None
        assert "control" in analysis
        assert "treatment" in analysis
        assert "improvement" in analysis
        assert "recommendation" in analysis


class TestONNXOptimizer:
    """Tests для ONNX Optimizer"""

    @pytest.mark.asyncio
    async def test_export_to_onnx(self, sample_model, tmp_path):
        """Test экспорта в ONNX"""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        optimizer = ONNXOptimizer()

        # Сохраним модель
        model_path = tmp_path / "model.pt"
        torch.save(sample_model.state_dict(), model_path)

        # Export
        onnx_path = tmp_path / "model.onnx"
        success = await optimizer.export_to_onnx(
            model=sample_model,
            model_path=model_path,
            output_path=onnx_path,
            input_shape=(1, 60, 110)  # batch, timesteps, features
        )

        assert success
        assert onnx_path.exists()

    @pytest.mark.asyncio
    async def test_benchmark(self, sample_model, tmp_path):
        """Test benchmarking"""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        optimizer = ONNXOptimizer()

        # Export first
        model_path = tmp_path / "model.pt"
        torch.save(sample_model.state_dict(), model_path)

        onnx_path = tmp_path / "model.onnx"
        await optimizer.export_to_onnx(
            model=sample_model,
            model_path=model_path,
            output_path=onnx_path,
            input_shape=(1, 60, 110)
        )

        # Benchmark
        metrics = await optimizer.benchmark(
            onnx_path=onnx_path,
            input_shape=(1, 60, 110),
            num_iterations=100,
            warmup_iterations=10
        )

        assert metrics is not None
        assert "latency_ms" in metrics
        assert "throughput" in metrics
        assert metrics["latency_ms"] > 0
        assert metrics["throughput"] > 0


@pytest.mark.asyncio
async def test_end_to_end_workflow(tmp_path):
    """
    End-to-end test всего workflow:
    1. Register model
    2. Export to ONNX
    3. Create A/B test
    4. Simulate predictions
    5. Analyze results
    """
    # 1. Create и register model
    registry = ModelRegistry(registry_dir=str(tmp_path / "models"))

    model = HybridCNNLSTM(
        input_size=110,
        lstm_hidden_size=64,
        lstm_layers=1,
        cnn_channels=[32],
        kernel_sizes=[3],
        dropout=0.1
    )

    model_path = tmp_path / "model_v1.pt"
    torch.save(model.state_dict(), model_path)

    await registry.register_model(
        name="hybrid_model",
        version="1.0.0",
        model_path=model_path,
        model_type="HybridCNNLSTM",
        metrics={"accuracy": 0.80}
    )

    await registry.promote_to_production("hybrid_model", "1.0.0")

    # 2. Register v2 (improved)
    model_v2_path = tmp_path / "model_v2.pt"
    torch.save(model.state_dict(), model_v2_path)

    await registry.register_model(
        name="hybrid_model",
        version="1.1.0",
        model_path=model_v2_path,
        model_type="HybridCNNLSTM",
        metrics={"accuracy": 0.85}
    )

    await registry.set_model_stage("hybrid_model", "1.1.0", ModelStage.STAGING)

    # 3. Create A/B test
    ab_manager = ABTestManager()

    await ab_manager.create_experiment(
        experiment_id="v1_vs_v2",
        control_model_name="hybrid_model",
        control_model_version="1.0.0",
        treatment_model_name="hybrid_model",
        treatment_model_version="1.1.0",
        control_traffic=0.9,
        treatment_traffic=0.1
    )

    # 4. Simulate predictions
    for i in range(200):
        variant = await ab_manager.route_traffic("v1_vs_v2")

        # Treatment is better
        if variant == ModelVariant.TREATMENT:
            outcome = PredictionOutcome(
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                prediction={"direction": "BUY"},
                actual_outcome=0.01 if i % 4 != 0 else -0.01,  # 75% win
                latency_ms=3.0
            )
        else:
            outcome = PredictionOutcome(
                timestamp=datetime.now(),
                symbol="BTCUSDT",
                prediction={"direction": "BUY"},
                actual_outcome=0.01 if i % 2 == 0 else -0.01,  # 50% win
                latency_ms=4.0
            )

        await ab_manager.record_prediction("v1_vs_v2", variant, outcome)

    # 5. Analyze
    analysis = await ab_manager.analyze_experiment("v1_vs_v2")

    assert analysis is not None
    assert analysis["treatment"]["samples"] > 0
    assert analysis["control"]["samples"] > 0

    # Treatment should be better
    improvement = analysis["improvement"]
    assert improvement["latency_ms"] < 0  # Lower latency
    # Note: accuracy comparison might not be significant with small sample

    print(f"Analysis: {analysis['recommendation']}")
