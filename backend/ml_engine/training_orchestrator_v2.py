#!/usr/bin/env python3
"""
Training Orchestrator v2 - Industry Standard Integration.

Главный модуль, который объединяет все компоненты ML системы:
- Загрузка данных (Feature Store или legacy)
- Балансировка классов
- Модель HybridCNNLSTM v2
- Оптимизированный Trainer
- MLflow tracking
- Checkpoint management
- Автоматический выбор оптимальных параметров

Путь: backend/ml_engine/training_orchestrator_v2.py
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field, asdict
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OrchestratorConfig:
    """
    Конфигурация Training Orchestrator.
    
    Объединяет все параметры для запуска обучения.
    """
    
    # === Data Source ===
    use_feature_store: bool = True
    feature_store_days: int = 30  # Дней данных
    fallback_to_legacy: bool = True
    legacy_storage_path: str = "data/ml_training"
    
    # === Symbols ===
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    auto_select_symbols: bool = False
    max_symbols: int = 50
    
    # === Model ===
    model_preset: str = "production_small"  # production_small, production_large, quick_experiment
    
    # === Training ===
    trainer_preset: str = "production_small"
    
    # === Balancing ===
    balancing_preset: str = "production"  # production, conservative, aggressive
    
    # === Output ===
    output_dir: str = "models/trained"
    experiment_name: str = "ml_training"
    
    # === MLflow ===
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "mlruns"
    
    # === Device ===
    device: str = "auto"  # auto, cuda, cpu
    
    # === Reproducibility ===
    seed: int = 42
    
    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ============================================================================
# TRAINING ORCHESTRATOR V2
# ============================================================================

class TrainingOrchestratorV2:
    """
    Training Orchestrator v2 - главный модуль управления обучением.
    
    Workflow:
    1. Загрузка данных (Feature Store или legacy)
    2. Анализ данных и оптимизация параметров
    3. Балансировка классов
    4. Создание модели
    5. Обучение с оптимизированными параметрами
    6. Оценка и сохранение модели
    7. Логирование в MLflow
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Args:
            config: Конфигурация оркестратора
        """
        self.config = config or OrchestratorConfig()
        self.device = self.config.get_device()
        
        # Устанавливаем seed
        self._set_seed(self.config.seed)
        
        # Создаём директории
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Компоненты (инициализируются при запуске)
        self.model = None
        self.trainer = None
        self.data_loader = None
        self.balancing_strategy = None
        
        # Метаданные обучения
        self.training_metadata: Dict[str, Any] = {}
        
        logger.info(
            f"✓ TrainingOrchestratorV2 initialized:\n"
            f"  • Device: {self.device}\n"
            f"  • Feature Store: {self.config.use_feature_store}\n"
            f"  • Model preset: {self.config.model_preset}\n"
            f"  • Trainer preset: {self.config.trainer_preset}\n"
            f"  • Output: {self.output_dir}"
        )
    
    def _set_seed(self, seed: int):
        """Установить seed для воспроизводимости."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    async def run_training(
        self,
        symbols: Optional[List[str]] = None,
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Запустить полный цикл обучения.
        
        Args:
            symbols: Список символов (если None - из config)
            days: Количество дней данных (если None - из config)
        
        Returns:
            Dict с результатами обучения
        """
        symbols = symbols or self.config.symbols
        days = days or self.config.feature_store_days
        
        logger.info("\n" + "=" * 80)
        logger.info("🚀 ЗАПУСК ОБУЧЕНИЯ ML МОДЕЛИ")
        logger.info("=" * 80)
        logger.info(f"Символы: {symbols}")
        logger.info(f"Период данных: {days} дней")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 80 + "\n")
        
        start_time = datetime.now()
        
        try:
            # 1. Загрузка данных
            logger.info("📥 Этап 1/6: Загрузка данных...")
            train_loader, val_loader, test_loader, data_stats = await self._load_data(
                symbols, days
            )
            
            if train_loader is None:
                raise ValueError("Не удалось загрузить данные для обучения")
            
            # 2. Анализ данных
            logger.info("\n📊 Этап 2/6: Анализ данных...")
            data_analysis = self._analyze_data(train_loader, val_loader)
            
            # 3. Инициализация балансировки
            logger.info("\n⚖️ Этап 3/6: Настройка балансировки классов...")
            self._setup_balancing(data_analysis)
            
            # 4. Создание модели
            logger.info("\n🏗️ Этап 4/6: Создание модели...")
            self._create_model(data_analysis)
            
            # 5. Обучение
            logger.info("\n🎓 Этап 5/6: Обучение модели...")
            training_results = self._train_model(train_loader, val_loader, test_loader)
            
            # 6. Сохранение и логирование
            logger.info("\n💾 Этап 6/6: Сохранение результатов...")
            save_results = self._save_model(training_results)
            
            # Формируем итоговый отчёт
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            results = {
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_time_seconds': total_time,
                'data_stats': data_stats,
                'data_analysis': data_analysis,
                'training_results': training_results,
                'save_results': save_results,
                'model_path': save_results.get('model_path'),
                'best_metrics': {
                    'val_loss': self.trainer.best_val_loss if self.trainer else None,
                    'val_f1': self.trainer.best_val_f1 if self.trainer else None
                }
            }
            
            self._log_final_results(results)
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении: {e}")
            logger.exception("Детали ошибки:")
            
            return {
                'status': 'error',
                'error': str(e),
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
    
    async def _load_data(
        self,
        symbols: List[str],
        days: int
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], Dict]:
        """Загрузка данных из Feature Store или legacy."""
        
        data_stats = {
            'symbols': symbols,
            'days': days,
            'source': None,
            'total_samples': 0
        }
        
        # Попытка загрузки из Feature Store
        if self.config.use_feature_store:
            try:
                logger.info("Попытка загрузки из Feature Store...")
                
                from backend.ml_engine.feature_store.feature_store import get_feature_store
                from backend.ml_engine.feature_store.feature_schema import DEFAULT_SCHEMA
                from backend.ml_engine.training.data_loader import HistoricalDataLoader, DataConfig
                
                feature_store = get_feature_store()
                await feature_store.initialize()
                
                # Получаем данные
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                features_df = await feature_store.get_training_data(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if features_df is not None and len(features_df) > 0:
                    logger.info(f"✓ Загружено {len(features_df)} записей из Feature Store")
                    
                    # Конвертируем в DataLoaders
                    data_config = DataConfig(
                        batch_size=256,  # Оптимизированный batch size
                        sequence_length=60
                    )
                    
                    data_loader = HistoricalDataLoader(data_config)
                    
                    train_loader, val_loader, test_loader = data_loader.load_from_dataframe(
                        features_df=features_df,
                        feature_columns=DEFAULT_SCHEMA.get_all_feature_columns(),
                        label_column=DEFAULT_SCHEMA.label_column,
                        timestamp_column=DEFAULT_SCHEMA.timestamp_column,
                        apply_resampling=True
                    )
                    
                    data_stats['source'] = 'feature_store'
                    data_stats['total_samples'] = len(features_df)
                    
                    return train_loader, val_loader, test_loader, data_stats
                
            except Exception as e:
                logger.warning(f"Feature Store недоступен: {e}")
        
        # Fallback к legacy
        if self.config.fallback_to_legacy:
            logger.info("Загрузка из legacy storage...")
            
            try:
                from backend.ml_engine.training.data_loader import HistoricalDataLoader, DataConfig
                
                data_config = DataConfig(
                    storage_path=self.config.legacy_storage_path,
                    batch_size=256,
                    sequence_length=60
                )
                
                data_loader = HistoricalDataLoader(data_config)
                train_loader, val_loader, test_loader = data_loader.load_and_split(
                    symbols=symbols
                )
                
                if train_loader is not None:
                    data_stats['source'] = 'legacy'
                    data_stats['total_samples'] = len(train_loader.dataset)
                    
                    logger.info(f"✓ Загружено из legacy: {data_stats['total_samples']} samples")
                    
                    return train_loader, val_loader, test_loader, data_stats
                
            except Exception as e:
                logger.error(f"Legacy загрузка не удалась: {e}")
        
        return None, None, None, data_stats
    
    def _analyze_data(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, Any]:
        """Анализ данных для оптимизации параметров."""
        
        # Собираем все labels
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch['label'].numpy())
        
        all_labels = np.array(all_labels)
        
        from collections import Counter
        class_dist = Counter(all_labels)
        
        # Вычисляем статистики
        total_samples = len(all_labels)
        num_classes = len(class_dist)
        
        # Imbalance ratio
        max_count = max(class_dist.values())
        min_count = min(class_dist.values())
        imbalance_ratio = max_count / max(min_count, 1)
        
        # Соотношение samples/params
        # Примерная оценка параметров модели
        estimated_params = self._estimate_model_params()
        samples_per_param = total_samples / max(estimated_params, 1)
        
        analysis = {
            'total_samples': total_samples,
            'num_classes': num_classes,
            'class_distribution': dict(class_dist),
            'imbalance_ratio': imbalance_ratio,
            'estimated_params': estimated_params,
            'samples_per_param': samples_per_param,
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'recommendations': []
        }
        
        # Рекомендации
        if imbalance_ratio > 3:
            analysis['recommendations'].append(
                f"Высокий imbalance ({imbalance_ratio:.1f}x). "
                "Рекомендуется Focal Loss с gamma>=2.5 и oversampling."
            )
        
        if samples_per_param < 50:
            analysis['recommendations'].append(
                f"Мало данных на параметр ({samples_per_param:.1f}). "
                "Рекомендуется уменьшить модель и увеличить регуляризацию."
            )
        
        if total_samples < 10000:
            analysis['recommendations'].append(
                "Малый датасет (<10K samples). "
                "Рекомендуется использовать preset 'production_small'."
            )
        
        logger.info(f"\n📊 Анализ данных:")
        logger.info(f"  • Total samples: {total_samples:,}")
        logger.info(f"  • Class distribution: {dict(class_dist)}")
        logger.info(f"  • Imbalance ratio: {imbalance_ratio:.2f}x")
        logger.info(f"  • Samples per param: {samples_per_param:.1f}")
        
        if analysis['recommendations']:
            logger.info(f"\n📋 Рекомендации:")
            for rec in analysis['recommendations']:
                logger.info(f"  • {rec}")
        
        return analysis
    
    def _estimate_model_params(self) -> int:
        """Оценка количества параметров модели."""
        preset = self.config.model_preset
        
        if preset == "production_small":
            return 150000  # ~150K params
        elif preset == "production_large":
            return 500000  # ~500K params
        elif preset == "quick_experiment":
            return 50000   # ~50K params
        else:
            return 200000  # Default estimate
    
    def _setup_balancing(self, data_analysis: Dict[str, Any]):
        """Настройка стратегии балансировки."""
        
        from backend.ml_engine.training.class_balancing_v2 import (
            create_balancing_strategy,
            ClassBalancingConfigV2,
            BalancingMethod
        )
        
        # Выбираем preset на основе анализа
        imbalance = data_analysis.get('imbalance_ratio', 1.0)
        
        if imbalance > 5:
            preset = "aggressive"
        elif imbalance > 2:
            preset = "production"
        else:
            preset = "conservative"
        
        # Override если указан в config
        preset = self.config.balancing_preset or preset
        
        self.balancing_strategy = create_balancing_strategy(preset)
        
        logger.info(f"✓ Balancing strategy: {preset}")
    
    def _create_model(self, data_analysis: Dict[str, Any]):
        """Создание модели на основе анализа данных."""
        
        from backend.ml_engine.models.hybrid_cnn_lstm_v2 import (
            create_model_v2_from_preset
        )
        
        preset = self.config.model_preset
        
        # Автоматический выбор preset на основе данных
        samples = data_analysis.get('total_samples', 0)
        
        if samples < 20000 and preset == "auto":
            preset = "production_small"
        elif samples >= 20000 and preset == "auto":
            preset = "production_large"
        
        self.model = create_model_v2_from_preset(preset)
        self.model.to(self.device)
        
        # Логируем информацию о модели
        model_size = self.model.get_model_size()
        logger.info(f"✓ Model created: {preset}")
        logger.info(f"  • Total params: {model_size['total_params']:,}")
        logger.info(f"  • Trainable params: {model_size['trainable_params']:,}")
    
    def _train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader]
    ) -> Dict[str, Any]:
        """Обучение модели."""
        
        from backend.ml_engine.training.model_trainer_v2 import (
            create_trainer_v2,
            TrainerConfigV2
        )
        
        # Создаём trainer с оптимизированными параметрами
        trainer_config = TrainerConfigV2(
            epochs=150,
            learning_rate=5e-5,
            batch_size=256,
            weight_decay=0.01,
            label_smoothing=0.1,
            use_augmentation=True,
            mixup_alpha=0.2,
            focal_gamma=2.5,
            early_stopping_patience=20,
            checkpoint_dir=str(self.output_dir / "checkpoints"),
            device=str(self.device)
        )
        
        self.trainer = create_trainer_v2(self.model, trainer_config)
        
        # Обучение
        history = self.trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # Формируем результаты
        results = {
            'epochs_trained': len(history),
            'best_val_loss': self.trainer.best_val_loss,
            'best_val_f1': self.trainer.best_val_f1,
            'final_metrics': history[-1].to_dict() if history else {},
            'history': [m.to_dict() for m in history]
        }
        
        return results
    
    def _save_model(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Сохранение модели и метаданных."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"hybrid_cnn_lstm_v2_{timestamp}"
        model_path = self.output_dir / f"{model_name}.pt"
        
        # Сохраняем модель
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'training_results': training_results,
            'timestamp': timestamp
        }, model_path)
        
        # Сохраняем метаданные
        metadata_path = self.output_dir / f"{model_name}_metadata.json"
        
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'config': asdict(self.config),
            'training_results': {
                'epochs_trained': training_results['epochs_trained'],
                'best_val_loss': training_results['best_val_loss'],
                'best_val_f1': training_results['best_val_f1']
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"✓ Model saved: {model_path}")
        logger.info(f"✓ Metadata saved: {metadata_path}")
        
        return {
            'model_path': str(model_path),
            'metadata_path': str(metadata_path),
            'model_name': model_name
        }
    
    def _log_final_results(self, results: Dict[str, Any]):
        """Логирование финальных результатов."""
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО")
        logger.info("=" * 80)
        
        if results['status'] == 'success':
            logger.info(f"✅ Статус: Успех")
            logger.info(f"⏱️ Время: {results['total_time_seconds']:.1f} секунд")
            
            if results.get('best_metrics'):
                logger.info(f"\n📊 Лучшие метрики:")
                logger.info(f"  • Val Loss: {results['best_metrics']['val_loss']:.4f}")
                logger.info(f"  • Val F1: {results['best_metrics']['val_f1']:.4f}")
            
            if results.get('model_path'):
                logger.info(f"\n💾 Модель сохранена: {results['model_path']}")
        else:
            logger.error(f"❌ Статус: Ошибка")
            logger.error(f"Ошибка: {results.get('error', 'Unknown')}")
        
        logger.info("=" * 80 + "\n")


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Точка входа для CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Training Orchestrator v2")
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="Торговые пары для обучения"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Количество дней данных"
    )
    parser.add_argument(
        "--model-preset",
        choices=["production_small", "production_large", "quick_experiment"],
        default="production_small",
        help="Пресет модели"
    )
    parser.add_argument(
        "--output-dir",
        default="models/trained",
        help="Директория для сохранения моделей"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Устройство для обучения"
    )
    
    args = parser.parse_args()
    
    # Создаём конфигурацию
    config = OrchestratorConfig(
        symbols=args.symbols,
        feature_store_days=args.days,
        model_preset=args.model_preset,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Запускаем обучение
    orchestrator = TrainingOrchestratorV2(config)
    results = await orchestrator.run_training()
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
