#!/usr/bin/env python3
"""
Тест интеграции v2 компонентов.

Проверяет что все v2 классы используются в training_orchestrator.py.
"""

import sys
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_v2_imports():
    """Проверка что v2 классы импортируются."""
    print("=" * 80)
    print("ТЕСТ 1: Проверка импортов v2 компонентов")
    print("=" * 80)

    try:
        # Импортируем из training_orchestrator
        from backend.ml_engine.training_orchestrator import (
            HybridCNNLSTM,
            ModelConfig,
            ModelTrainer,
            TrainerConfig
        )

        # Проверяем что это v2 версии
        print(f"✅ HybridCNNLSTM импортирован: {HybridCNNLSTM.__module__}")
        assert "v2" in HybridCNNLSTM.__module__, "HybridCNNLSTM должен быть из v2 модуля!"

        print(f"✅ ModelConfig импортирован: {ModelConfig.__module__}")
        assert "v2" in ModelConfig.__module__, "ModelConfig должен быть из v2 модуля!"

        print(f"✅ ModelTrainer импортирован: {ModelTrainer.__module__}")
        assert "v2" in ModelTrainer.__module__, "ModelTrainer должен быть из v2 модуля!"

        print(f"✅ TrainerConfig импортирован: {TrainerConfig.__module__}")
        assert "v2" in TrainerConfig.__module__, "TrainerConfig должен быть из v2 модуля!"

        print("\n✅ ВСЕ v2 КОМПОНЕНТЫ ИМПОРТИРУЮТСЯ КОРРЕКТНО!\n")
        return True

    except ImportError as e:
        print(f"❌ ОШИБКА ИМПОРТА: {e}")
        return False
    except AssertionError as e:
        print(f"❌ ОШИБКА: {e}")
        return False


def test_v2_orchestrator_available():
    """Проверка что TrainingOrchestratorV2 доступен."""
    print("=" * 80)
    print("ТЕСТ 2: Проверка доступности TrainingOrchestratorV2")
    print("=" * 80)

    try:
        from backend.ml_engine.training_orchestrator import (
            TrainingOrchestratorV2,
            OrchestratorConfig
        )

        print(f"✅ TrainingOrchestratorV2 доступен: {TrainingOrchestratorV2.__module__}")
        print(f"✅ OrchestratorConfig доступен: {OrchestratorConfig.__module__}")

        # Проверяем что это действительно v2
        assert "v2" in TrainingOrchestratorV2.__module__, "Должен быть v2 orchestrator!"

        print("\n✅ TrainingOrchestratorV2 ДОСТУПЕН ДЛЯ ИСПОЛЬЗОВАНИЯ!\n")
        return True

    except ImportError as e:
        print(f"⚠️  TrainingOrchestratorV2 не найден: {e}")
        print("⚠️  Это нормально если файл training_orchestrator_v2.py не существует")
        return False
    except AssertionError as e:
        print(f"❌ ОШИБКА: {e}")
        return False


def test_trainer_config_v2_params():
    """Проверка что TrainerConfig имеет v2 параметры."""
    print("=" * 80)
    print("ТЕСТ 3: Проверка оптимизированных параметров v2")
    print("=" * 80)

    try:
        from backend.ml_engine.training_orchestrator import TrainerConfig

        # Создаем конфигурацию с дефолтными параметрами
        config = TrainerConfig()

        # Проверяем критичные параметры
        print(f"Learning Rate: {config.learning_rate}")
        assert config.learning_rate == 5e-5, f"Learning rate должен быть 5e-5, получен {config.learning_rate}"
        print("✅ Learning rate = 5e-5 (оптимизирован)")

        print(f"Weight Decay: {config.weight_decay}")
        assert config.weight_decay == 0.01, f"Weight decay должен быть 0.01, получен {config.weight_decay}"
        print("✅ Weight decay = 0.01 (оптимизирован)")

        print(f"Epochs: {config.epochs}")
        assert config.epochs == 150, f"Epochs должен быть 150, получен {config.epochs}"
        print("✅ Epochs = 150 (оптимизирован)")

        print(f"LR Scheduler: {config.lr_scheduler}")
        assert config.lr_scheduler == "CosineAnnealingWarmRestarts", f"Scheduler должен быть CosineAnnealingWarmRestarts"
        print("✅ LR Scheduler = CosineAnnealingWarmRestarts (v2)")

        # Проверяем новые v2 параметры
        assert hasattr(config, 'label_smoothing'), "Должен быть параметр label_smoothing"
        print(f"✅ Label Smoothing = {config.label_smoothing} (v2 feature)")

        assert hasattr(config, 'use_augmentation'), "Должен быть параметр use_augmentation"
        print(f"✅ Use Augmentation = {config.use_augmentation} (v2 feature)")

        print("\n✅ ВСЕ ОПТИМИЗИРОВАННЫЕ ПАРАМЕТРЫ v2 ПРИСУТСТВУЮТ!\n")
        return True

    except AssertionError as e:
        print(f"❌ ОШИБКА: {e}")
        return False
    except Exception as e:
        print(f"❌ НЕОЖИДАННАЯ ОШИБКА: {e}")
        return False


def test_model_config_v2_params():
    """Проверка что ModelConfig имеет оптимизированные параметры."""
    print("=" * 80)
    print("ТЕСТ 4: Проверка оптимизированной архитектуры модели")
    print("=" * 80)

    try:
        from backend.ml_engine.training_orchestrator import ModelConfig

        config = ModelConfig()

        print(f"CNN Channels: {config.cnn_channels}")
        assert config.cnn_channels == (32, 64, 128), f"CNN channels должны быть (32, 64, 128)"
        print("✅ CNN channels = (32, 64, 128) - оптимизировано для малого датасета")

        print(f"LSTM Hidden: {config.lstm_hidden}")
        assert config.lstm_hidden == 128, f"LSTM hidden должен быть 128"
        print("✅ LSTM hidden = 128 - оптимизировано для малого датасета")

        print(f"Dropout: {config.dropout}")
        assert config.dropout == 0.4, f"Dropout должен быть 0.4"
        print("✅ Dropout = 0.4 - увеличен для регуляризации")

        print("\n✅ АРХИТЕКТУРА МОДЕЛИ ОПТИМИЗИРОВАНА!\n")
        return True

    except AssertionError as e:
        print(f"❌ ОШИБКА: {e}")
        return False
    except Exception as e:
        print(f"❌ НЕОЖИДАННАЯ ОШИБКА: {e}")
        return False


def main():
    """Запуск всех тестов."""
    print("\n")
    print("PROVERKA INTEGRACII V2 KOMPONENTOV")
    print("=" * 80)
    print()

    results = []

    # Тест 1: Импорты v2
    results.append(("v2 Imports", test_v2_imports()))

    # Тест 2: TrainingOrchestratorV2
    results.append(("TrainingOrchestratorV2 Available", test_v2_orchestrator_available()))

    # Тест 3: TrainerConfig параметры
    results.append(("TrainerConfig v2 Parameters", test_trainer_config_v2_params()))

    # Тест 4: ModelConfig параметры
    results.append(("ModelConfig v2 Parameters", test_model_config_v2_params()))

    # Итоги
    print("=" * 80)
    print("ИТОГИ ТЕСТОВ")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Успешно: {passed}/{total}")

    if passed == total:
        print("\nVSE TESTY PROJDENY! V2 KOMPONENTY INTEGRIROVANY KORREKTNO!")
        return 0
    else:
        print(f"\n{total - passed} test(ov) provaleny")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
