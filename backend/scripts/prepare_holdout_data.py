#!/usr/bin/env python3
"""
Скрипт для подготовки holdout данных для ML бэктестинга.

Использование:
    python -m backend.scripts.prepare_holdout_data --days 30 --output data/holdout/test_data.npz

Параметры:
    --days: Количество дней данных (по умолчанию 30)
    --output: Путь для сохранения файла (по умолчанию data/holdout/test_data.npz)
    --sequence-length: Длина последовательности (по умолчанию 60)
    --symbol: Торговая пара (по умолчанию BTCUSDT)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.core.logger import get_logger

logger = get_logger(__name__)


def prepare_holdout_from_feature_store(
    days: int = 30,
    sequence_length: int = 60,
    output_path: str = "data/holdout/test_data.npz"
) -> bool:
    """
    Подготовка holdout данных из Feature Store.

    Args:
        days: Количество дней данных
        sequence_length: Длина последовательности для модели
        output_path: Путь для сохранения

    Returns:
        True если успешно, False при ошибке
    """
    try:
        from backend.ml_engine.feature_store.feature_store import get_feature_store

        logger.info(f"Загрузка данных из Feature Store за последние {days} дней...")

        feature_store = get_feature_store()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = feature_store.read_offline_features(
            feature_group="training_features",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        if df is None or df.empty:
            logger.error("Feature Store вернул пустые данные")
            return False

        logger.info(f"Загружено {len(df)} записей из Feature Store")

        return _process_and_save(df, sequence_length, output_path)

    except Exception as e:
        logger.error(f"Ошибка загрузки из Feature Store: {e}")
        return False


def prepare_holdout_from_parquet(
    parquet_path: str,
    sequence_length: int = 60,
    output_path: str = "data/holdout/test_data.npz"
) -> bool:
    """
    Подготовка holdout данных из Parquet файла.

    Args:
        parquet_path: Путь к Parquet файлу с features
        sequence_length: Длина последовательности
        output_path: Путь для сохранения

    Returns:
        True если успешно
    """
    try:
        logger.info(f"Загрузка данных из {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        logger.info(f"Загружено {len(df)} записей")

        return _process_and_save(df, sequence_length, output_path)

    except Exception as e:
        logger.error(f"Ошибка загрузки из Parquet: {e}")
        return False


def _process_and_save(
    df: pd.DataFrame,
    sequence_length: int,
    output_path: str
) -> bool:
    """
    Обработка DataFrame и сохранение в NPZ формат.
    """
    try:
        # Определяем колонку с метками
        label_column = None
        for col in ['future_direction_60s', 'future_direction', 'label', 'target']:
            if col in df.columns:
                label_column = col
                break

        if label_column is None:
            logger.error(f"Не найдена колонка с метками. Доступные колонки: {list(df.columns)}")
            return False

        logger.info(f"Используется колонка меток: {label_column}")

        # Определяем колонку timestamp
        timestamp_column = None
        for col in ['timestamp', 'time', 'datetime', 'date']:
            if col in df.columns:
                timestamp_column = col
                break

        # Определяем колонку цены
        price_column = None
        for col in ['close', 'price', 'close_price']:
            if col in df.columns:
                price_column = col
                break

        # Получаем feature колонки (исключаем служебные)
        exclude_cols = {
            label_column, timestamp_column, price_column,
            'symbol', 'id', 'index', 'open', 'high', 'low', 'volume',
            'future_return_60s', 'future_price_60s'
        }
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

        logger.info(f"Найдено {len(feature_cols)} feature колонок")

        if len(feature_cols) == 0:
            logger.error("Не найдено feature колонок")
            return False

        # Удаляем NaN значения
        df = df.dropna(subset=feature_cols + [label_column])
        logger.info(f"После удаления NaN: {len(df)} записей")

        if len(df) < sequence_length * 2:
            logger.error(f"Недостаточно данных: {len(df)} < {sequence_length * 2}")
            return False

        # Извлекаем данные
        features = df[feature_cols].values.astype(np.float32)
        labels = df[label_column].values.astype(np.int64)

        timestamps = None
        if timestamp_column:
            timestamps = pd.to_datetime(df[timestamp_column]).values

        prices = None
        if price_column:
            prices = df[price_column].values.astype(np.float32)

        # Создаём последовательности
        logger.info(f"Создание последовательностей длиной {sequence_length}...")

        n_samples = len(features) - sequence_length + 1
        X = np.zeros((n_samples, sequence_length, len(feature_cols)), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            X[i] = features[i:i + sequence_length]
            y[i] = labels[i + sequence_length - 1]  # Метка для последнего элемента последовательности

        # Timestamps и prices для последнего элемента каждой последовательности
        if timestamps is not None:
            timestamps = timestamps[sequence_length - 1:]
        if prices is not None:
            prices = prices[sequence_length - 1:]

        logger.info(f"Создано {len(X)} последовательностей")
        logger.info(f"Shape X: {X.shape}, Shape y: {y.shape}")

        # Проверяем распределение классов
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Распределение классов: {dict(zip(unique, counts))}")

        # Создаём директорию если нужно
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Сохраняем
        save_dict = {'X': X, 'y': y}
        if timestamps is not None:
            save_dict['timestamps'] = timestamps
        if prices is not None:
            save_dict['prices'] = prices
        save_dict['feature_names'] = np.array(feature_cols)

        np.savez_compressed(output_path, **save_dict)

        file_size = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Данные сохранены в {output_path} ({file_size:.2f} MB)")

        return True

    except Exception as e:
        logger.error(f"Ошибка обработки данных: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Подготовка holdout данных для ML бэктестинга")
    parser.add_argument("--days", type=int, default=30, help="Количество дней данных")
    parser.add_argument("--output", type=str, default="data/holdout/test_data.npz", help="Путь для сохранения")
    parser.add_argument("--sequence-length", type=int, default=60, help="Длина последовательности")
    parser.add_argument("--parquet", type=str, help="Путь к Parquet файлу (вместо Feature Store)")

    args = parser.parse_args()

    print("=" * 60)
    print("Подготовка Holdout данных для ML Backtesting")
    print("=" * 60)

    if args.parquet:
        success = prepare_holdout_from_parquet(
            parquet_path=args.parquet,
            sequence_length=args.sequence_length,
            output_path=args.output
        )
    else:
        success = prepare_holdout_from_feature_store(
            days=args.days,
            sequence_length=args.sequence_length,
            output_path=args.output
        )

    if success:
        print("\n✅ Holdout данные успешно подготовлены!")
        print(f"   Файл: {args.output}")
        print("\nТеперь вы можете использовать 'Holdout Set' в ML Backtesting.")
    else:
        print("\n❌ Ошибка подготовки данных")
        print("   Попробуйте использовать 'Feature Store' как источник данных в UI")
        sys.exit(1)


if __name__ == "__main__":
    main()
