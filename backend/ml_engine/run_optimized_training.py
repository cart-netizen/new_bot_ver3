#!/usr/bin/env python3
"""
Скрипт запуска обучения ML модели с оптимизированными параметрами.

Использование:
    python run_optimized_training.py --symbols BTCUSDT ETHUSDT --days 30
    python run_optimized_training.py --preset quick  # Быстрый тест
    python run_optimized_training.py --preset production  # Production

Путь: backend/ml_engine/run_optimized_training.py
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Добавляем корневую директорию в путь
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np


def setup_environment():
    """Настройка окружения перед запуском."""
    # Устанавливаем seed для воспроизводимости
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Отключаем предупреждения
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)


def get_preset_config(preset: str) -> dict:
    """Получить конфигурацию для пресета."""
    
    presets = {
        'quick': {
            'description': 'Быстрый тест (5 минут)',
            'epochs': 10,
            'learning_rate': 1e-4,
            'batch_size': 128,
            'model_preset': 'quick_experiment',
            'early_stopping_patience': 5
        },
        'development': {
            'description': 'Разработка (30 минут)',
            'epochs': 50,
            'learning_rate': 5e-5,
            'batch_size': 256,
            'model_preset': 'production_small',
            'early_stopping_patience': 10
        },
        'production': {
            'description': 'Production (2-4 часа)',
            'epochs': 150,
            'learning_rate': 5e-5,
            'batch_size': 256,
            'model_preset': 'production_small',
            'early_stopping_patience': 20
        },
        'production_large': {
            'description': 'Production Large Data (4-8 часов)',
            'epochs': 100,
            'learning_rate': 1e-4,
            'batch_size': 128,
            'model_preset': 'production_large',
            'early_stopping_patience': 15
        }
    }
    
    if preset not in presets:
        print(f"❌ Неизвестный пресет: {preset}")
        print(f"Доступные пресеты: {list(presets.keys())}")
        sys.exit(1)
    
    return presets[preset]


def print_banner():
    """Вывод баннера."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       🤖 ML MODEL TRAINING - OPTIMIZED v2                                   ║
║                                                                              ║
║       Industry Standard Configuration                                        ║
║       - Learning Rate: 5e-5 (not 0.001!)                                    ║
║       - Batch Size: 256                                                      ║
║       - Focal Loss + Label Smoothing                                         ║
║       - MixUp Data Augmentation                                              ║
║       - CosineAnnealingWarmRestarts                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config(args, preset_config: dict):
    """Вывод конфигурации."""
    print("\n📋 КОНФИГУРАЦИЯ:")
    print("─" * 60)
    print(f"  Пресет: {args.preset} - {preset_config['description']}")
    print(f"  Символы: {args.symbols}")
    print(f"  Дней данных: {args.days}")
    print(f"  Epochs: {preset_config['epochs']}")
    print(f"  Learning Rate: {preset_config['learning_rate']}")
    print(f"  Batch Size: {preset_config['batch_size']}")
    print(f"  Model: {preset_config['model_preset']}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")
    print("─" * 60)


async def run_training_with_orchestrator(args, preset_config: dict):
    """Запуск обучения через Orchestrator."""
    
    from backend.ml_engine.training_orchestrator_v2 import (
        TrainingOrchestratorV2,
        OrchestratorConfig
    )
    
    config = OrchestratorConfig(
        symbols=args.symbols,
        feature_store_days=args.days,
        model_preset=preset_config['model_preset'],
        trainer_preset=args.preset,
        output_dir=args.output_dir,
        device=args.device,
        use_feature_store=not args.legacy_only
    )
    
    orchestrator = TrainingOrchestratorV2(config)
    results = await orchestrator.run_training()
    
    return results


async def run_training_standalone(args, preset_config: dict):
    """
    Standalone обучение без полного Orchestrator.
    
    Для случаев когда нужен более простой запуск.
    """
    from backend.core.logger import get_logger
    logger = get_logger(__name__)
    
    logger.info("Standalone training mode")
    
    # 1. Загрузка данных
    logger.info("\n📥 Загрузка данных...")
    
    from backend.ml_engine.training.data_loader import HistoricalDataLoader, DataConfig
    
    data_config = DataConfig(
        storage_path="data/ml_training",
        batch_size=preset_config['batch_size'],
        sequence_length=60
    )
    
    data_loader = HistoricalDataLoader(data_config)
    
    try:
        train_loader, val_loader, test_loader = data_loader.load_and_split(
            symbols=args.symbols
        )
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return {'status': 'error', 'error': str(e)}
    
    if train_loader is None:
        logger.error("Нет данных для обучения!")
        return {'status': 'error', 'error': 'No training data'}
    
    logger.info(f"✓ Данные загружены: {len(train_loader.dataset)} samples")
    
    # 2. Создание модели
    logger.info("\n🏗️ Создание модели...")
    
    from backend.ml_engine.models.hybrid_cnn_lstm_v2 import create_model_v2_from_preset
    
    model = create_model_v2_from_preset(preset_config['model_preset'])
    
    device = torch.device(args.device if args.device != 'auto' else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(device)
    
    model_size = model.get_model_size()
    logger.info(f"✓ Модель создана: {model_size['total_params']:,} параметров")
    
    # 3. Обучение
    logger.info("\n🎓 Обучение модели...")
    
    from backend.ml_engine.training.model_trainer_v2 import (
        ModelTrainerV2,
        TrainerConfigV2
    )
    
    trainer_config = TrainerConfigV2(
        epochs=preset_config['epochs'],
        learning_rate=preset_config['learning_rate'],
        batch_size=preset_config['batch_size'],
        weight_decay=0.01,
        label_smoothing=0.1,
        use_augmentation=True,
        mixup_alpha=0.2,
        focal_gamma=2.5,
        early_stopping_patience=preset_config['early_stopping_patience'],
        checkpoint_dir=args.output_dir,
        device=str(device)
    )
    
    trainer = ModelTrainerV2(model, trainer_config)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    # 4. Сохранение
    logger.info("\n💾 Сохранение модели...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"model_{timestamp}.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_loss': trainer.best_val_loss,
        'best_val_f1': trainer.best_val_f1,
        'history': [m.to_dict() for m in history]
    }, model_path)
    
    logger.info(f"✓ Модель сохранена: {model_path}")
    
    return {
        'status': 'success',
        'model_path': str(model_path),
        'best_val_loss': trainer.best_val_loss,
        'best_val_f1': trainer.best_val_f1,
        'epochs_trained': len(history)
    }


def main():
    """Главная функция."""
    
    parser = argparse.ArgumentParser(
        description="Обучение ML модели с оптимизированными параметрами",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s --preset quick                    # Быстрый тест
  %(prog)s --preset production               # Production обучение
  %(prog)s --symbols BTCUSDT ETHUSDT --days 30  # Указать символы
  %(prog)s --legacy-only                     # Только legacy данные
        """
    )
    
    parser.add_argument(
        "--preset",
        choices=["quick", "development", "production", "production_large"],
        default="production",
        help="Пресет конфигурации (default: production)"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Торговые пары (default: BTCUSDT ETHUSDT)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Количество дней данных (default: 30)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="models/trained",
        help="Директория для сохранения (default: models/trained)"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Устройство (default: auto)"
    )
    
    parser.add_argument(
        "--legacy-only",
        action="store_true",
        help="Использовать только legacy данные"
    )
    
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Standalone режим без Orchestrator"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать конфигурацию без запуска"
    )
    
    args = parser.parse_args()
    
    # Настройка окружения
    setup_environment()
    
    # Получаем конфигурацию пресета
    preset_config = get_preset_config(args.preset)
    
    # Выводим баннер и конфигурацию
    print_banner()
    print_config(args, preset_config)
    
    if args.dry_run:
        print("\n⚠️ Dry run - обучение не запущено")
        return 0
    
    # Запуск обучения
    print("\n🚀 Запуск обучения...")
    print("=" * 60)
    
    try:
        if args.standalone:
            results = asyncio.run(run_training_standalone(args, preset_config))
        else:
            results = asyncio.run(run_training_with_orchestrator(args, preset_config))
        
        # Вывод результатов
        print("\n" + "=" * 60)
        
        if results.get('status') == 'success':
            print("✅ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
            print(f"   Model: {results.get('model_path', 'N/A')}")
            print(f"   Best Val Loss: {results.get('best_val_loss', 'N/A'):.4f}")
            print(f"   Best Val F1: {results.get('best_val_f1', 'N/A'):.4f}")
            return 0
        else:
            print(f"❌ ОШИБКА: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Обучение прервано пользователем")
        return 130
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
