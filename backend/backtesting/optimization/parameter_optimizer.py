"""
Parameter Optimizer - оптимизация параметров стратегий для бэктестинга.

Методы оптимизации:
1. Grid Search - перебор всех комбинаций параметров
2. Random Search - случайная выборка параметров
3. Genetic Algorithm - эволюционная оптимизация (будущее)

Использование:
    optimizer = ParameterOptimizer(base_config)

    # Grid search
    param_grid = {
        "stop_loss_pct": [1.0, 2.0, 3.0],
        "take_profit_pct": [2.0, 3.0, 4.0],
        "position_size_pct": [5.0, 10.0, 15.0]
    }

    results = await optimizer.grid_search(param_grid, metric="sharpe_ratio")
    best_params = results[0]["parameters"]
"""

import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
import itertools
import random
import copy

from backend.core.logger import get_logger
from backend.backtesting.models import BacktestConfig, BacktestResult, RiskConfig, StrategyConfig
from backend.backtesting.core.backtesting_engine import BacktestingEngine
from backend.backtesting.core.data_handler import HistoricalDataHandler
from backend.backtesting.core.simulated_exchange import SimulatedExchange

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """Результат одной итерации оптимизации."""
    parameters: Dict[str, Any]
    backtest_result: BacktestResult
    metric_value: float  # Значение оптимизируемой метрики
    rank: int = 0  # Ранг в отсортированных результатах


@dataclass
class OptimizationReport:
    """Отчет об оптимизации."""
    method: str  # "grid_search", "random_search", "genetic"
    total_iterations: int
    completed_iterations: int
    failed_iterations: int
    best_result: Optional[OptimizationResult] = None
    all_results: List[OptimizationResult] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


class ParameterOptimizer:
    """
    Оптимизатор параметров стратегий бэктестинга.

    Запускает множественные бэктесты с разными параметрами
    и находит оптимальную конфигурацию по заданной метрике.
    """

    def __init__(
        self,
        base_config: BacktestConfig,
        data_handler: HistoricalDataHandler,
        simulated_exchange: SimulatedExchange,
        max_workers: Optional[int] = None
    ):
        """
        Args:
            base_config: Базовая конфигурация бэктеста (будет клонирована)
            data_handler: Handler для исторических данных
            simulated_exchange: Симулятор биржи
            max_workers: Максимум параллельных процессов (None = CPU count)
        """
        self.base_config = base_config
        self.data_handler = data_handler
        self.simulated_exchange = simulated_exchange
        self.max_workers = max_workers or mp.cpu_count()

        logger.info(
            f"ParameterOptimizer инициализирован: max_workers={self.max_workers}"
        )

    async def grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
        top_n: int = 10,
        parallel: bool = True
    ) -> List[OptimizationResult]:
        """
        Grid Search - перебор всех комбинаций параметров.

        Args:
            param_grid: Сетка параметров {param_name: [values]}
            metric: Метрика для оптимизации (sharpe_ratio, total_return_pct, etc.)
            top_n: Вернуть топ-N лучших результатов
            parallel: Запускать бэктесты параллельно

        Returns:
            List[OptimizationResult]: Отсортированные результаты (лучшие первые)
        """
        started_at = datetime.now()

        logger.info(f"🔍 Grid Search запущен: metric={metric}, parallel={parallel}")
        logger.info(f"Параметры для оптимизации: {list(param_grid.keys())}")

        # Генерируем все комбинации параметров
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        total_iterations = len(all_combinations)
        logger.info(f"Всего комбинаций: {total_iterations}")

        # Запускаем бэктесты
        results: List[OptimizationResult] = []
        failed_count = 0

        if parallel and total_iterations > 1:
            # Параллельный запуск
            logger.info(f"Параллельное выполнение с {self.max_workers} workers")
            results, failed_count = await self._run_parallel_backtests(
                all_combinations, param_names, metric
            )
        else:
            # Последовательный запуск
            logger.info("Последовательное выполнение")
            results, failed_count = await self._run_sequential_backtests(
                all_combinations, param_names, metric
            )

        # Сортируем по метрике (лучшие первые)
        results.sort(key=lambda x: x.metric_value, reverse=True)

        # Проставляем ранги
        for i, result in enumerate(results, 1):
            result.rank = i

        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()

        logger.info(
            f"✅ Grid Search завершен: {len(results)} успешных, {failed_count} неудачных, "
            f"duration={duration:.1f}s"
        )

        if results:
            best = results[0]
            logger.info(
                f"🏆 Лучший результат: {metric}={best.metric_value:.4f}, "
                f"params={best.parameters}"
            )

        return results[:top_n]

    async def random_search(
        self,
        param_ranges: Dict[str, tuple],
        n_iterations: int = 50,
        metric: str = "sharpe_ratio",
        top_n: int = 10,
        parallel: bool = True
    ) -> List[OptimizationResult]:
        """
        Random Search - случайная выборка параметров.

        Более эффективен чем grid search для больших пространств параметров.

        Args:
            param_ranges: Диапазоны параметров {param_name: (min, max)}
            n_iterations: Количество случайных итераций
            metric: Метрика для оптимизации
            top_n: Вернуть топ-N лучших результатов
            parallel: Запускать параллельно

        Returns:
            List[OptimizationResult]: Отсортированные результаты
        """
        started_at = datetime.now()

        logger.info(f"🔍 Random Search запущен: n_iterations={n_iterations}, metric={metric}")

        # Генерируем случайные комбинации параметров
        param_names = list(param_ranges.keys())
        all_combinations = []

        for _ in range(n_iterations):
            combination = []
            for param_name in param_names:
                min_val, max_val = param_ranges[param_name]

                # Определяем тип параметра по типу границ
                if isinstance(min_val, int) and isinstance(max_val, int):
                    value = random.randint(min_val, max_val)
                elif isinstance(min_val, float) or isinstance(max_val, float):
                    value = random.uniform(min_val, max_val)
                else:
                    # Fallback на uniform для неизвестных типов
                    value = random.uniform(float(min_val), float(max_val))

                combination.append(value)

            all_combinations.append(tuple(combination))

        # Запускаем бэктесты
        if parallel and n_iterations > 1:
            results, failed_count = await self._run_parallel_backtests(
                all_combinations, param_names, metric
            )
        else:
            results, failed_count = await self._run_sequential_backtests(
                all_combinations, param_names, metric
            )

        # Сортируем
        results.sort(key=lambda x: x.metric_value, reverse=True)

        for i, result in enumerate(results, 1):
            result.rank = i

        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()

        logger.info(
            f"✅ Random Search завершен: {len(results)} успешных, {failed_count} неудачных, "
            f"duration={duration:.1f}s"
        )

        if results:
            best = results[0]
            logger.info(
                f"🏆 Лучший результат: {metric}={best.metric_value:.4f}, "
                f"params={best.parameters}"
            )

        return results[:top_n]

    async def _run_sequential_backtests(
        self,
        all_combinations: List[tuple],
        param_names: List[str],
        metric: str
    ) -> Tuple[List[OptimizationResult], int]:
        """Запуск бэктестов последовательно."""
        results: List[OptimizationResult] = []
        failed_count = 0

        for i, combination in enumerate(all_combinations, 1):
            params = dict(zip(param_names, combination))

            logger.info(f"[{i}/{len(all_combinations)}] Запуск с параметрами: {params}")

            try:
                # Клонируем конфиг и применяем параметры
                config = self._apply_parameters(self.base_config, params)

                # Запускаем бэктест
                engine = BacktestingEngine(
                    config=config,
                    data_handler=self.data_handler,
                    simulated_exchange=copy.deepcopy(self.simulated_exchange)
                )

                backtest_result = await engine.run()

                # Извлекаем метрику
                metric_value = self._extract_metric(backtest_result, metric)

                results.append(OptimizationResult(
                    parameters=params,
                    backtest_result=backtest_result,
                    metric_value=metric_value
                ))

            except Exception as e:
                logger.error(f"Ошибка в итерации {i}: {e}")
                failed_count += 1

        return results, failed_count

    async def _run_parallel_backtests(
        self,
        all_combinations: List[tuple],
        param_names: List[str],
        metric: str
    ) -> Tuple[List[OptimizationResult], int]:
        """Запуск бэктестов параллельно."""
        results: List[OptimizationResult] = []
        failed_count = 0

        # Запускаем батчами по max_workers
        batch_size = self.max_workers

        for batch_start in range(0, len(all_combinations), batch_size):
            batch_end = min(batch_start + batch_size, len(all_combinations))
            batch = all_combinations[batch_start:batch_end]

            logger.info(
                f"Запуск батча {batch_start//batch_size + 1}: "
                f"итерации {batch_start+1}-{batch_end} из {len(all_combinations)}"
            )

            # Запускаем batch параллельно
            batch_tasks = []
            for combination in batch:
                params = dict(zip(param_names, combination))
                task = self._run_single_backtest(params, metric)
                batch_tasks.append(task)

            # Ждем завершения batch
            # Type: List[Union[OptimizationResult, BaseException]]
            batch_results: List[Union[Optional[OptimizationResult], BaseException]] = await asyncio.gather(
                *batch_tasks, return_exceptions=True
            )

            # Собираем результаты (фильтруем только успешные OptimizationResult)
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Ошибка в бэктесте: {result}")
                    failed_count += 1
                elif result is not None:
                    results.append(result)
                else:
                    failed_count += 1

        return results, failed_count

    async def _run_single_backtest(
        self,
        params: Dict[str, Any],
        metric: str
    ) -> Optional[OptimizationResult]:
        """Запуск одного бэктеста."""
        try:
            # Клонируем конфиг и применяем параметры
            config = self._apply_parameters(self.base_config, params)

            # Запускаем бэктест
            engine = BacktestingEngine(
                config=config,
                data_handler=self.data_handler,
                simulated_exchange=copy.deepcopy(self.simulated_exchange)
            )

            backtest_result = await engine.run()

            # Извлекаем метрику
            metric_value = self._extract_metric(backtest_result, metric)

            return OptimizationResult(
                parameters=params,
                backtest_result=backtest_result,
                metric_value=metric_value
            )

        except Exception as e:
            logger.error(f"Ошибка бэктеста с params={params}: {e}")
            return None

    def _apply_parameters(
        self,
        base_config: BacktestConfig,
        params: Dict[str, Any]
    ) -> BacktestConfig:
        """
        Применить параметры к конфигурации.

        Поддерживает nested параметры через "." (dot notation):
        - "risk_config.stop_loss_pct"
        - "strategy_config.min_strategies_for_signal"
        """
        # Глубокое копирование конфига
        config = copy.deepcopy(base_config)

        for param_path, value in params.items():
            # Поддержка dot notation для nested objects
            if "." in param_path:
                parts = param_path.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(config, param_path, value)

        return config

    def _extract_metric(
        self,
        backtest_result: BacktestResult,
        metric: str
    ) -> float:
        """Извлечь значение метрики из результата бэктеста."""
        # Специальная обработка для некоторых метрик
        if metric == "total_return":
            return backtest_result.total_pnl

        if metric == "total_return_pct":
            return backtest_result.total_pnl_pct

        # Остальные метрики из metrics объекта
        if hasattr(backtest_result.metrics, metric):
            value = getattr(backtest_result.metrics, metric)
            return value if value is not None else -float('inf')

        logger.warning(f"Неизвестная метрика: {metric}, используем total_return_pct")
        return backtest_result.total_pnl_pct
