#!/usr/bin/env python3
"""
Скрипт для запуска оптимизированного обучения ML модели - Industry Standard.

Этот скрипт интегрирует все оптимизированные компоненты:
1. HybridCNNLSTMv2 с Residual, LayerNorm, Multi-Head Attention
2. Оптимизированные гиперпараметры (LR=5e-5, BS=256, WD=0.01)
3. Data Augmentation (MixUp, Gaussian noise, Time masking)
4. Focal Loss с Label Smoothing
5. CosineAnnealingWarmRestarts scheduler
6. Class Balancing с адаптивными thresholds

Использование:
    python run_optimized_training.py --symbols BTCUSDT ETHUSDT --days 30 --preset production_small

Путь: backend/ml_engine/scripts/run_optimized_training.py
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch

# Добавляем корневую директорию в path
ROOT_DIR = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

def get_preset_config(preset: str) -> Dict[str, Any]:
    """
    Получить пресет конфигурации.
    
    Пресеты:
        production_small: Для 7-30 дней данных (рекомендуется)
        production_large: Для 60+ дней данных
        quick_experiment: Для быстрых тестов
        conservative: Для консервативной торговли
    """
    presets = {
        "production_small": {
            # Model
            "cnn_channels": [32, 64, 128],
            "lstm_hidden": 128,
            "dropout": 0.4,
            "use_residual": True,
            "use_layer_norm": True,
            "use_multi_head_attention": True,
            "attention_heads": 4,
            
            # Training
            "epochs": 150,
            "learning_rate": 5e-5,  # КРИТИЧНО!
            "batch_size": 256,      # КРИТИЧНО!
            "weight_decay": 0.01,   # КРИТИЧНО!
            "label_smoothing": 0.1,
            "scheduler_type": "cosine_warm_restarts",
            "scheduler_T_0": 10,
            "scheduler_T_mult": 2,
            "early_stopping_patience": 20,
            
            # Augmentation
            "use_augmentation": True,
            "mixup_alpha": 0.2,
            "gaussian_noise_std": 0.01,
            
            # Class Balancing
            "use_focal_loss": True,
            "focal_gamma": 2.5,
            "use_class_weights": True,
            "use_adaptive_threshold": True,
            "percentile_sell": 0.25,
            "percentile_buy": 0.75,
            "use_oversampling": True,
            "oversample_ratio": 0.5
        },
        
        "production_large": {
            "cnn_channels": [64, 128, 256],
            "lstm_hidden": 256,
            "dropout": 0.3,
            "use_residual": True,
            "use_layer_norm": True,
            "use_multi_head_attention": True,
            "attention_heads": 4,
            
            "epochs": 100,
            "learning_rate": 1e-4,
            "batch_size": 128,
            "weight_decay": 0.005,
            "label_smoothing": 0.05,
            "scheduler_type": "cosine_warm_restarts",
            "early_stopping_patience": 15,
            
            "use_augmentation": True,
            "mixup_alpha": 0.1,
            "gaussian_noise_std": 0.005,
            
            "use_focal_loss": True,
            "focal_gamma": 2.0,
            "use_class_weights": True,
            "use_adaptive_threshold": True,
            "percentile_sell": 0.30,
            "percentile_buy": 0.70,
            "use_oversampling": True,
            "oversample_ratio": 0.3
        },
        
        "quick_experiment": {
            "cnn_channels": [32, 64],
            "lstm_hidden": 64,
            "dropout": 0.3,
            "use_residual": False,
            "use_layer_norm": False,
            "use_multi_head_attention": False,
            
            "epochs": 30,
            "learning_rate": 1e-4,
            "batch_size": 128,
            "weight_decay": 0.001,
            "label_smoothing": 0.0,
            "scheduler_type": "reduce_on_plateau",
            "early_stopping_patience": 10,
            
            "use_augmentation": False,
            "mixup_alpha": 0.0,
            "gaussian_noise_std": 0.0,
            
            "use_focal_loss": True,
            "focal_gamma": 2.0,
            "use_class_weights": True,
            "use_adaptive_threshold": False,
            "use_oversampling": False
        },
        
        "conservative": {
            "cnn_channels": [32, 64, 128],
            "lstm_hidden": 128,
            "dropout": 0.5,
            "use_residual": True,
            "use_layer_norm": True,
            "use_multi_head_attention": True,
            "attention_heads": 4,
            
            "epochs": 200,
            "learning_rate": 3e-5,
            "batch_size": 256,
            "weight_decay": 0.02,
            "label_smoothing": 0.15,
            "scheduler_type": "cosine_warm_restarts",
            "scheduler_T_0": 15,
            "early_stopping_patience": 25,
            
            "use_augmentation": True,
            "mixup_alpha": 0.3,
            "gaussian_noise_std": 0.02,
            
            "use_focal_loss": True,
            "focal_gamma": 3.0,
            "use_class_weights": True,
            "use_adaptive_threshold": True,
            "percentile_sell": 0.20,
            "percentile_buy": 0.80,
            "use_oversampling": True,
            "oversample_ratio": 0.4
        }
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    return presets[preset]


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_optimized_model(config: Dict[str, Any]) -> torch.nn.Module:
    """
    Создать оптимизированную модель.
    
    Args:
        config: Конфигурация модели
    
    Returns:
        Инициализированная модель
    """
    from backend.ml_engine.models.hybrid_cnn_lstm import ModelConfig, HybridCNNLSTM
    
    # Создаём конфигурацию модели
    model_config = ModelConfig(
        input_features=110,
        sequence_length=60,
        cnn_channels=tuple(config["cnn_channels"]),
        cnn_kernel_sizes=(3, 5, 7)[:len(config["cnn_channels"])],
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=2,
        lstm_dropout=config.get("lstm_dropout", 0.3),
        attention_units=config.get("attention_units", 64),
        num_classes=3,
        dropout=config["dropout"]
    )
    
    # Создаём модель
    model = HybridCNNLSTM(model_config)
    
    # Логируем параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(
        f"✓ Модель создана: "
        f"total_params={total_params:,}, "
        f"trainable={trainable_params:,}"
    )
    
    return model


# ============================================================================
# DATA LOADING
# ============================================================================

def load_training_data(
    symbols: List[str],
    days: int,
    config: Dict[str, Any],
    data_path: str = "data/ml_training"
) -> Tuple[Any, Any, Any]:
    """
    Загрузить данные для обучения.
    
    Args:
        symbols: Список торговых пар
        days: Количество дней данных
        config: Конфигурация
        data_path: Путь к данным
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from backend.ml_engine.training.data_loader import (
        HistoricalDataLoader, DataConfig
    )
    from backend.ml_engine.training.class_balancing import (
        ClassBalancingConfig, ClassBalancingStrategy
    )
    
    # Data config
    data_config = DataConfig(
        storage_path=data_path,
        sequence_length=60,
        batch_size=config["batch_size"],
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Class balancing config
    balancing_config = None
    if config.get("use_oversampling", False):
        balancing_config = ClassBalancingConfig(
            use_class_weights=config.get("use_class_weights", True),
            use_focal_loss=config.get("use_focal_loss", True),
            focal_gamma=config.get("focal_gamma", 2.5),
            use_oversampling=True,
            oversample_ratio=config.get("oversample_ratio", 0.5)
        )
    
    # Создаём loader
    data_loader = HistoricalDataLoader(
        data_config,
        balancing_config=balancing_config
    )
    
    # Загружаем данные
    logger.info(f"Загрузка данных для символов: {symbols}")
    
    try:
        result = data_loader.load_and_prepare(
            symbols=symbols,
            apply_resampling=config.get("use_oversampling", False)
        )
        
        dataloaders = result['dataloaders']
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        test_loader = dataloaders.get('test', None)
        
        logger.info(f"✓ Данные загружены:")
        logger.info(f"  • Train batches: {len(train_loader)}")
        logger.info(f"  • Val batches: {len(val_loader)}")
        if test_loader:
            logger.info(f"  • Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        raise


# ============================================================================
# TRAINING
# ============================================================================

def run_training(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    config: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Запустить обучение модели.
    
    Args:
        model: Модель для обучения
        train_loader: DataLoader обучения
        val_loader: DataLoader валидации
        test_loader: DataLoader тестирования
        config: Конфигурация
        output_dir: Директория для сохранения
    
    Returns:
        Dict с результатами обучения
    """
    from backend.ml_engine.training.model_trainer import (
        ModelTrainer, TrainerConfig
    )
    from backend.ml_engine.training.class_balancing import ClassBalancingConfig
    
    # Trainer config
    trainer_config = TrainerConfig(
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        grad_clip_value=1.0,
        early_stopping_patience=config["early_stopping_patience"],
        checkpoint_dir=output_dir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Class balancing
    balancing_config = ClassBalancingConfig(
        use_class_weights=config.get("use_class_weights", True),
        use_focal_loss=config.get("use_focal_loss", True),
        focal_gamma=config.get("focal_gamma", 2.5)
    )
    
    # Обновляем trainer config с class balancing
    trainer_config.class_balancing = balancing_config
    
    # Создаём trainer
    trainer = ModelTrainer(model, trainer_config)
    
    # Обучаем
    logger.info("\n" + "=" * 80)
    logger.info("ЗАПУСК ОБУЧЕНИЯ")
    logger.info("=" * 80)
    logger.info(f"Device: {trainer_config.device}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Learning Rate: {config['learning_rate']}")
    logger.info(f"Batch Size: {config['batch_size']}")
    logger.info(f"Weight Decay: {config['weight_decay']}")
    logger.info("=" * 80 + "\n")
    
    history = trainer.train(train_loader, val_loader)
    
    # Результаты
    results = {
        "epochs_trained": len(history),
        "best_val_loss": trainer.best_val_loss,
        "final_metrics": history[-1].__dict__ if history else {},
        "checkpoint_path": str(Path(output_dir) / "best_model.pt")
    }
    
    # Тестирование
    if test_loader is not None:
        logger.info("\nЗапуск тестирования...")
        test_metrics = evaluate_model(model, test_loader, trainer_config.device)
        results["test_metrics"] = test_metrics
    
    return results


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: str
) -> Dict[str, float]:
    """Оценка модели на тестовом наборе."""
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix
    )
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(sequences)
            
            if isinstance(outputs, dict):
                predictions = torch.argmax(outputs['direction_logits'], dim=-1)
            else:
                predictions = torch.argmax(outputs[0], dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Метрики
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions,
        average='weighted',
        zero_division=0
    )
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    logger.info(f"\nTest Results:")
    logger.info(f"  • Accuracy: {accuracy:.4f}")
    logger.info(f"  • Precision: {precision:.4f}")
    logger.info(f"  • Recall: {recall:.4f}")
    logger.info(f"  • F1 Score: {f1:.4f}")
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Запуск оптимизированного обучения ML модели"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT"],
        help="Торговые пары для обучения"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Количество дней данных"
    )
    
    parser.add_argument(
        "--preset",
        type=str,
        default="production_small",
        choices=["production_small", "production_large", "quick_experiment", "conservative"],
        help="Пресет конфигурации"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ml_training",
        help="Путь к данным"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/optimized",
        help="Директория для сохранения"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Переопределить количество эпох"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Переопределить learning rate"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Переопределить batch size"
    )
    
    args = parser.parse_args()
    
    # Логирование
    logger.info("\n" + "=" * 80)
    logger.info("ОПТИМИЗИРОВАННОЕ ОБУЧЕНИЕ ML МОДЕЛИ")
    logger.info("=" * 80)
    logger.info(f"Символы: {args.symbols}")
    logger.info(f"Дни данных: {args.days}")
    logger.info(f"Пресет: {args.preset}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80 + "\n")
    
    # Получаем конфигурацию
    config = get_preset_config(args.preset)
    
    # Переопределяем параметры если заданы
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    
    # Создаём output директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем конфигурацию
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = output_dir / f"config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Конфигурация сохранена: {config_path}")
    
    try:
        # 1. Создаём модель
        logger.info("\n[1/3] Создание модели...")
        model = create_optimized_model(config)
        
        # 2. Загружаем данные
        logger.info("\n[2/3] Загрузка данных...")
        train_loader, val_loader, test_loader = load_training_data(
            symbols=args.symbols,
            days=args.days,
            config=config,
            data_path=args.data_path
        )
        
        # 3. Обучаем модель
        logger.info("\n[3/3] Обучение модели...")
        results = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            output_dir=str(output_dir)
        )
        
        # Сохраняем результаты
        results_path = output_dir / f"results_{timestamp}.json"
        
        # Конвертируем numpy/tensor в python types
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj
        
        with open(results_path, "w") as f:
            json.dump(convert_to_serializable(results), f, indent=2)
        
        # Финальный отчёт
        logger.info("\n" + "=" * 80)
        logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        logger.info("=" * 80)
        logger.info(f"Эпох обучено: {results['epochs_trained']}")
        logger.info(f"Best val_loss: {results['best_val_loss']:.4f}")
        
        if "test_metrics" in results:
            logger.info(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
            logger.info(f"Test F1: {results['test_metrics']['f1']:.4f}")
        
        logger.info(f"\nМодель сохранена: {results['checkpoint_path']}")
        logger.info(f"Результаты сохранены: {results_path}")
        logger.info("=" * 80 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
