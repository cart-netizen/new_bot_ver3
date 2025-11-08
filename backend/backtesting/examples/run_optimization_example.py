"""
Пример использования ParameterOptimizer для поиска оптимальных параметров.

Демонстрирует:
1. Grid Search - перебор всех комбинаций
2. Random Search - случайная выборка
3. Параллелизацию бэктестов
4. Кэширование данных

Запуск:
    python3 -m backend.backtesting.examples.run_optimization_example
"""

import asyncio
from datetime import datetime, timedelta
import sys
import os

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from backend.backtesting.models import BacktestConfig, RiskConfig, StrategyConfig
from backend.backtesting.core.data_handler import HistoricalDataHandler
from backend.backtesting.core.simulated_exchange import SimulatedExchange
from backend.backtesting.optimization import ParameterOptimizer
from backend.core.logger import get_logger

logger = get_logger(__name__)


async def run_grid_search_example():
    """Пример Grid Search оптимизации."""
    print("=" * 80)
    print("GRID SEARCH OPTIMIZATION EXAMPLE")
    print("=" * 80)

    # Базовая конфигурация (будет клонирована для каждого теста)
    base_config = BacktestConfig(
        name="BTCUSDT Grid Search",
        symbol="BTCUSDT",

        # Короткий период для быстрого теста
        start_date=datetime(2024, 11, 1),
        end_date=datetime(2024, 11, 7),  # 1 неделя
        candle_interval="1m",

        # Капитал
        initial_capital=10000.0,

        # Risk config (будут оптимизированы)
        risk_config=RiskConfig(
            max_open_positions=3,
            position_size_pct=10.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0
        ),

        # Стратегии
        strategy_config=StrategyConfig(
            consensus_mode="weighted",
            min_strategies_for_signal=2,
            min_consensus_confidence=0.5
        ),

        # Оптимизации
        use_orderbook_data=True,
        use_market_trades=True,
        use_ml_model=False,  # Выключаем для скорости
        use_cache=True  # Включаем кэш!
    )

    # Инициализация компонентов
    data_handler = HistoricalDataHandler(
        data_source="bybit",  # или другой источник
        cache_enabled=True
    )

    simulated_exchange = SimulatedExchange(
        initial_balance=base_config.initial_capital,
        commission_rate=0.001  # 0.1%
    )

    # Создаем оптимизатор
    optimizer = ParameterOptimizer(
        base_config=base_config,
        data_handler=data_handler,
        simulated_exchange=simulated_exchange,
        max_workers=4  # 4 параллельных процесса
    )

    # Сетка параметров для оптимизации
    param_grid = {
        "risk_config.stop_loss_pct": [1.0, 2.0, 3.0],
        "risk_config.take_profit_pct": [2.0, 4.0, 6.0],
        "risk_config.position_size_pct": [5.0, 10.0, 15.0],
        "strategy_config.min_strategies_for_signal": [2, 3, 4]
    }

    print(f"\nПараметры для оптимизации:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")

    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)

    print(f"\nВсего комбинаций: {total_combinations}")
    print("Запуск Grid Search...\n")

    # Запускаем Grid Search
    results = await optimizer.grid_search(
        param_grid=param_grid,
        metric="sharpe_ratio",  # Оптимизируем Sharpe Ratio
        top_n=10,  # Топ-10 лучших
        parallel=True  # Параллельный запуск
    )

    # Выводим результаты
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ GRID SEARCH")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n#{i} - Sharpe Ratio: {result.metric_value:.4f}")
        print(f"Параметры:")
        for param, value in result.parameters.items():
            print(f"  {param}: {value}")

        metrics = result.backtest_result.metrics
        print(f"Метрики:")
        print(f"  Total Return: ${result.backtest_result.total_pnl:.2f} ({result.backtest_result.total_pnl_pct:.2f}%)")
        print(f"  Trades: {metrics.total_trades}")
        print(f"  Win Rate: {metrics.win_rate_pct:.2f}%")
        print(f"  Profit Factor: {metrics.profit_factor:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        print(f"  Sortino Ratio: {metrics.sortino_ratio:.4f}")
        print(f"  Calmar Ratio: {metrics.calmar_ratio:.4f}")

    print("\n" + "=" * 80)


async def run_random_search_example():
    """Пример Random Search оптимизации."""
    print("\n" + "=" * 80)
    print("RANDOM SEARCH OPTIMIZATION EXAMPLE")
    print("=" * 80)

    # Аналогичная конфигурация
    base_config = BacktestConfig(
        name="BTCUSDT Random Search",
        symbol="BTCUSDT",
        start_date=datetime(2024, 11, 1),
        end_date=datetime(2024, 11, 7),
        candle_interval="1m",
        initial_capital=10000.0,
        risk_config=RiskConfig(
            max_open_positions=3,
            position_size_pct=10.0,
            stop_loss_pct=2.0,
            take_profit_pct=4.0
        ),
        strategy_config=StrategyConfig(
            consensus_mode="weighted",
            min_strategies_for_signal=2,
            min_consensus_confidence=0.5
        ),
        use_orderbook_data=True,
        use_market_trades=True,
        use_ml_model=False,
        use_cache=True
    )

    data_handler = HistoricalDataHandler(
        data_source="bybit",
        cache_enabled=True
    )

    simulated_exchange = SimulatedExchange(
        initial_balance=base_config.initial_capital,
        commission_rate=0.001
    )

    optimizer = ParameterOptimizer(
        base_config=base_config,
        data_handler=data_handler,
        simulated_exchange=simulated_exchange,
        max_workers=4
    )

    # Диапазоны параметров (непрерывные значения)
    param_ranges = {
        "risk_config.stop_loss_pct": (0.5, 5.0),
        "risk_config.take_profit_pct": (1.0, 10.0),
        "risk_config.position_size_pct": (5.0, 20.0),
        "strategy_config.min_consensus_confidence": (0.3, 0.8)
    }

    print(f"\nПараметры для оптимизации (диапазоны):")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"  {param}: [{min_val}, {max_val}]")

    print(f"\nИтераций: 20")
    print("Запуск Random Search...\n")

    # Запускаем Random Search
    results = await optimizer.random_search(
        param_ranges=param_ranges,
        n_iterations=20,  # 20 случайных комбинаций
        metric="total_return_pct",  # Оптимизируем доходность
        top_n=5,
        parallel=True
    )

    # Выводим результаты
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ RANDOM SEARCH")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n#{i} - Total Return: {result.metric_value:.2f}%")
        print(f"Параметры:")
        for param, value in result.parameters.items():
            print(f"  {param}: {value:.3f}")

        metrics = result.backtest_result.metrics
        print(f"Метрики:")
        print(f"  Total PnL: ${result.backtest_result.total_pnl:.2f}")
        print(f"  Trades: {metrics.total_trades}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
        print(f"  Win Rate: {metrics.win_rate_pct:.2f}%")

    print("\n" + "=" * 80)


async def main():
    """Главная функция."""
    try:
        # Grid Search
        await run_grid_search_example()

        # Random Search
        # await run_random_search_example()

    except KeyboardInterrupt:
        print("\n\nОптимизация прервана пользователем")
    except Exception as e:
        logger.error(f"Ошибка оптимизации: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
