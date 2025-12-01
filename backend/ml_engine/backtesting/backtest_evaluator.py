#!/usr/bin/env python3
"""
Backtest Evaluator - оценка ML модели на исторических данных.

Функциональность:
- Walk-forward backtesting (правильный метод для time series)
- Расчет метрик по периодам
- Симуляция P&L
- Анализ по классам (BUY/HOLD/SELL)

ВАЖНО: Бэктестинг нельзя делать на данных, использованных для обучения!
Используйте данные ПОСЛЕ периода обучения или отдельный holdout set.

Путь: backend/ml_engine/backtesting/backtest_evaluator.py
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Конфигурация бэктестинга."""
    # Данные
    sequence_length: int = 60
    batch_size: int = 128

    # Trading simulation
    initial_capital: float = 10000.0
    position_size: float = 0.1  # % от капитала на сделку
    commission: float = 0.001  # 0.1% комиссия
    slippage: float = 0.0005  # 0.05% проскальзывание

    # Confidence filtering
    min_confidence: float = 0.6  # Минимальный confidence для сделки
    use_confidence_filter: bool = True

    # Walk-forward settings
    n_periods: int = 5  # Количество периодов для walk-forward
    retrain_each_period: bool = False  # Переобучать модель каждый период

    # Output
    save_results: bool = True
    results_dir: str = "data/backtest_results"


@dataclass
class BacktestResults:
    """Результаты бэктестинга."""
    # Метрики классификации
    accuracy: float = 0.0
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1: Dict[str, float] = field(default_factory=dict)

    # Trading метрики
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # P&L метрики
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0

    # По периодам
    period_results: List[Dict] = field(default_factory=list)

    # Детали
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    actuals: np.ndarray = field(default_factory=lambda: np.array([]))
    confidences: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))


class BacktestEvaluator:
    """
    Оценщик модели на исторических данных.

    Использование:
        evaluator = BacktestEvaluator(model, config)
        results = evaluator.run_backtest(X_test, y_test, timestamps)
    """

    CLASS_NAMES = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[BacktestConfig] = None,
        device: str = "cuda"
    ):
        """
        Инициализация.

        Args:
            model: Обученная PyTorch модель
            config: Конфигурация бэктестинга
            device: Device для inference
        """
        self.model = model
        self.config = config or BacktestConfig()
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"BacktestEvaluator initialized on {self.device}")

    def run_backtest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None
    ) -> BacktestResults:
        """
        Запуск бэктестинга на данных.

        ВАЖНО: X, y должны быть данными, НЕ использованными при обучении!

        Args:
            X: Features (N, sequence_length, n_features)
            y: Labels (N,)
            timestamps: Временные метки (N,) - опционально
            prices: Цены для расчета P&L (N,) - опционально

        Returns:
            BacktestResults с метриками
        """
        tqdm.write("\n" + "=" * 70)
        tqdm.write("[BACKTEST] ЗАПУСК БЭКТЕСТИНГА")
        tqdm.write("=" * 70)
        tqdm.write(f"  • Samples: {len(X):,}")
        tqdm.write(f"  • Device: {self.device}")
        tqdm.write(f"  • Confidence filter: {self.config.min_confidence:.0%}")
        tqdm.write("=" * 70)

        # Получаем предсказания
        predictions, confidences = self._get_predictions(X)

        # Рассчитываем метрики классификации
        class_metrics = self._calculate_classification_metrics(
            predictions, y, confidences
        )

        # Рассчитываем trading метрики (если есть цены)
        if prices is not None:
            trading_metrics = self._calculate_trading_metrics(
                predictions, y, prices, confidences
            )
        else:
            trading_metrics = {}

        # Формируем результаты
        results = BacktestResults(
            accuracy=class_metrics['accuracy'],
            precision=class_metrics['precision'],
            recall=class_metrics['recall'],
            f1=class_metrics['f1'],
            total_trades=trading_metrics.get('total_trades', 0),
            winning_trades=trading_metrics.get('winning_trades', 0),
            losing_trades=trading_metrics.get('losing_trades', 0),
            win_rate=trading_metrics.get('win_rate', 0.0),
            total_pnl=trading_metrics.get('total_pnl', 0.0),
            total_pnl_percent=trading_metrics.get('total_pnl_percent', 0.0),
            max_drawdown=trading_metrics.get('max_drawdown', 0.0),
            sharpe_ratio=trading_metrics.get('sharpe_ratio', 0.0),
            profit_factor=trading_metrics.get('profit_factor', 0.0),
            predictions=predictions,
            actuals=y,
            confidences=confidences,
            timestamps=timestamps if timestamps is not None else np.array([])
        )

        # Логируем результаты
        self._log_results(results)

        return results

    def run_walk_forward_backtest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        prices: Optional[np.ndarray] = None
    ) -> BacktestResults:
        """
        Walk-forward бэктестинг - более строгая оценка.

        Разбивает данные на периоды и оценивает каждый отдельно.
        Это даёт более реалистичную оценку производительности.

        Args:
            X: Features (N, sequence_length, n_features)
            y: Labels (N,)
            timestamps: Временные метки (N,) - опционально
            prices: Цены для расчета P&L (N,) - опционально

        Returns:
            BacktestResults с агрегированными метриками и результатами по периодам
        """
        n_samples = len(X)
        period_size = n_samples // self.config.n_periods

        tqdm.write("\n" + "=" * 70)
        tqdm.write("[WALK-FORWARD BACKTEST]")
        tqdm.write("=" * 70)
        tqdm.write(f"  • Samples: {n_samples:,}")
        tqdm.write(f"  • Periods: {self.config.n_periods}")
        tqdm.write(f"  • Period size: ~{period_size:,}")
        tqdm.write("=" * 70)

        all_predictions = []
        all_actuals = []
        all_confidences = []
        period_results = []

        for i in range(self.config.n_periods):
            start_idx = i * period_size
            end_idx = min((i + 1) * period_size, n_samples)

            X_period = X[start_idx:end_idx]
            y_period = y[start_idx:end_idx]
            prices_period = prices[start_idx:end_idx] if prices is not None else None

            # Предсказания для периода
            preds, confs = self._get_predictions(X_period)

            all_predictions.append(preds)
            all_actuals.append(y_period)
            all_confidences.append(confs)

            # Метрики периода
            period_metrics = self._calculate_classification_metrics(
                preds, y_period, confs
            )

            period_result = {
                'period': i + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'samples': len(X_period),
                'accuracy': period_metrics['accuracy'],
                'f1_macro': np.mean(list(period_metrics['f1'].values())),
                'class_distribution': dict(Counter(y_period))
            }

            if prices_period is not None:
                trading = self._calculate_trading_metrics(
                    preds, y_period, prices_period, confs
                )
                period_result.update({
                    'pnl_percent': trading.get('total_pnl_percent', 0),
                    'win_rate': trading.get('win_rate', 0)
                })

            period_results.append(period_result)

            tqdm.write(
                f"  Period {i+1}: acc={period_metrics['accuracy']:.2%}, "
                f"f1={period_result['f1_macro']:.2%}, "
                f"dist={period_result['class_distribution']}"
            )

        # Агрегируем результаты
        all_predictions = np.concatenate(all_predictions)
        all_actuals = np.concatenate(all_actuals)
        all_confidences = np.concatenate(all_confidences)

        # Общие метрики
        total_metrics = self._calculate_classification_metrics(
            all_predictions, all_actuals, all_confidences
        )

        results = BacktestResults(
            accuracy=total_metrics['accuracy'],
            precision=total_metrics['precision'],
            recall=total_metrics['recall'],
            f1=total_metrics['f1'],
            period_results=period_results,
            predictions=all_predictions,
            actuals=all_actuals,
            confidences=all_confidences
        )

        tqdm.write("\n" + "=" * 70)
        tqdm.write(f"[ИТОГО] Accuracy: {results.accuracy:.2%}")
        tqdm.write(f"[ИТОГО] F1 (macro): {np.mean(list(results.f1.values())):.2%}")
        tqdm.write("=" * 70 + "\n")

        return results

    def _get_predictions(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Получить предсказания модели."""
        self.model.eval()

        all_preds = []
        all_confs = []

        # Батчами для экономии памяти
        n_samples = len(X)
        batch_size = self.config.batch_size

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_X = X[i:i + batch_size]
                batch_tensor = torch.FloatTensor(batch_X).to(self.device)

                outputs = self.model(batch_tensor)

                # Извлекаем logits
                if isinstance(outputs, dict):
                    logits = outputs.get('direction_logits', outputs.get('logits'))
                    confidence = outputs.get('confidence', None)
                else:
                    logits = outputs
                    confidence = None

                # Предсказания
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1).cpu().numpy()

                # Confidence (если есть) или max probability
                if confidence is not None:
                    confs = confidence.squeeze().cpu().numpy()
                else:
                    confs = probs.max(dim=-1).values.cpu().numpy()

                all_preds.append(preds)
                all_confs.append(confs)

        return np.concatenate(all_preds), np.concatenate(all_confs)

    def _calculate_classification_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        confidences: np.ndarray
    ) -> Dict[str, Any]:
        """Рассчитать метрики классификации."""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        # С фильтром по confidence
        if self.config.use_confidence_filter:
            mask = confidences >= self.config.min_confidence
            if mask.sum() > 0:
                filtered_preds = predictions[mask]
                filtered_actuals = actuals[mask]
            else:
                filtered_preds = predictions
                filtered_actuals = actuals
        else:
            filtered_preds = predictions
            filtered_actuals = actuals

        # Общая accuracy
        accuracy = accuracy_score(filtered_actuals, filtered_preds)

        # Per-class метрики
        precision, recall, f1, support = precision_recall_fscore_support(
            filtered_actuals, filtered_preds,
            labels=[0, 1, 2],
            zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision': {self.CLASS_NAMES[i]: p for i, p in enumerate(precision)},
            'recall': {self.CLASS_NAMES[i]: r for i, r in enumerate(recall)},
            'f1': {self.CLASS_NAMES[i]: f for i, f in enumerate(f1)},
            'support': {self.CLASS_NAMES[i]: s for i, s in enumerate(support)}
        }

    def _calculate_trading_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        prices: np.ndarray,
        confidences: np.ndarray
    ) -> Dict[str, float]:
        """
        Рассчитать trading метрики.

        Простая симуляция:
        - BUY (1): открываем long, закрываем через 1 период
        - SELL (2): открываем short, закрываем через 1 период
        - HOLD (0): не торгуем
        """
        capital = self.config.initial_capital
        position_value = capital * self.config.position_size
        commission = self.config.commission
        slippage = self.config.slippage

        trades = []
        equity_curve = [capital]

        for i in range(len(predictions) - 1):
            pred = predictions[i]
            conf = confidences[i]
            actual = actuals[i]

            # Фильтр по confidence
            if self.config.use_confidence_filter and conf < self.config.min_confidence:
                equity_curve.append(equity_curve[-1])
                continue

            # Пропускаем HOLD
            if pred == 0:
                equity_curve.append(equity_curve[-1])
                continue

            # Рассчитываем P&L
            entry_price = prices[i]
            exit_price = prices[i + 1]

            if pred == 1:  # BUY
                pnl_percent = (exit_price - entry_price) / entry_price
            else:  # SELL
                pnl_percent = (entry_price - exit_price) / entry_price

            # Учитываем комиссию и проскальзывание
            pnl_percent -= (commission + slippage) * 2  # вход и выход

            # P&L в деньгах
            pnl = position_value * pnl_percent

            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'BUY' if pred == 1 else 'SELL',
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'correct': (pred == actual)
            })

            # Обновляем капитал
            capital += pnl
            equity_curve.append(capital)

        # Метрики
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'total_pnl_percent': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'profit_factor': 0.0
            }

        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in trades if t['pnl'] <= 0)
        total_pnl = sum(t['pnl'] for t in trades)
        total_pnl_percent = total_pnl / self.config.initial_capital

        # Max drawdown
        equity_array = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_drawdown = drawdown.max()

        # Sharpe ratio (упрощенный)
        returns = np.diff(equity_array) / equity_array[:-1]
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / len(trades) if trades else 0,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor
        }

    def _log_results(self, results: BacktestResults):
        """Логировать результаты."""
        tqdm.write("\n" + "=" * 70)
        tqdm.write("[BACKTEST RESULTS]")
        tqdm.write("=" * 70)

        tqdm.write("\n📊 CLASSIFICATION METRICS:")
        tqdm.write(f"  • Accuracy: {results.accuracy:.2%}")
        tqdm.write(f"  • Precision: {results.precision}")
        tqdm.write(f"  • Recall: {results.recall}")
        tqdm.write(f"  • F1: {results.f1}")

        if results.total_trades > 0:
            tqdm.write("\n💰 TRADING METRICS:")
            tqdm.write(f"  • Total trades: {results.total_trades}")
            tqdm.write(f"  • Win rate: {results.win_rate:.2%}")
            tqdm.write(f"  • Total P&L: ${results.total_pnl:.2f} ({results.total_pnl_percent:.2%})")
            tqdm.write(f"  • Max drawdown: {results.max_drawdown:.2%}")
            tqdm.write(f"  • Sharpe ratio: {results.sharpe_ratio:.2f}")
            tqdm.write(f"  • Profit factor: {results.profit_factor:.2f}")

        # Распределение предсказаний
        pred_dist = Counter(results.predictions)
        actual_dist = Counter(results.actuals)
        tqdm.write("\n📈 DISTRIBUTIONS:")
        tqdm.write(f"  • Predictions: {dict(pred_dist)}")
        tqdm.write(f"  • Actuals: {dict(actual_dist)}")

        tqdm.write("=" * 70 + "\n")


def run_backtest_from_checkpoint(
    checkpoint_path: str,
    data_path: str,
    config: Optional[BacktestConfig] = None
) -> BacktestResults:
    """
    Запуск бэктестинга из сохраненной модели.

    Args:
        checkpoint_path: Путь к .pt файлу модели
        data_path: Путь к данным для бэктестинга
        config: Конфигурация

    Returns:
        BacktestResults
    """
    from backend.ml_engine.models.hybrid_cnn_lstm_v2 import HybridCNNLSTMv2, ModelConfigV2

    # Загружаем модель
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = ModelConfigV2(**checkpoint.get('model_config', {}))
    model = HybridCNNLSTMv2(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Загружаем данные
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    timestamps = data.get('timestamps', None)
    prices = data.get('prices', None)

    # Запускаем бэктестинг
    evaluator = BacktestEvaluator(model, config)
    return evaluator.run_walk_forward_backtest(X, y, timestamps, prices)


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========

if __name__ == "__main__":
    print("BacktestEvaluator module loaded")
    print("Usage:")
    print("  evaluator = BacktestEvaluator(model, config)")
    print("  results = evaluator.run_backtest(X_test, y_test)")
    print("  results = evaluator.run_walk_forward_backtest(X_test, y_test)")
