"""
Полная реализация Dynamic Time Warping для анализа корреляций.

DTW позволяет находить схожесть временных рядов с учетом временного лага
и нелинейного растяжения/сжатия.

Путь: backend/strategy/correlation/dtw_calculator.py
"""
import numpy as np
from typing import Tuple, Optional
from numba import jit

from core.logger import get_logger

logger = get_logger(__name__)

# Попытка импорта numba для ускорения
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Заглушка декоратора если numba недоступна
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    NUMBA_AVAILABLE = False
    logger.warning("Numba недоступна, DTW будет медленнее")


class DTWCalculator:
    """
    Полная реализация Dynamic Time Warping.

    Использует dynamic programming для нахождения оптимального
    выравнивания между двумя временными рядами.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        distance_metric: str = 'euclidean',
        normalize: bool = True,
        step_pattern: str = 'symmetric2'
    ):
        """
        Инициализация DTW калькулятора.

        Args:
            window_size: Размер окна Sakoe-Chiba (ограничение поиска)
                        None = без ограничений
            distance_metric: Метрика расстояния (euclidean, manhattan)
            normalize: Нормализовать временные ряды перед сравнением
            step_pattern: Паттерн шагов (symmetric1, symmetric2, asymmetric)
        """
        self.window_size = window_size
        self.distance_metric = distance_metric
        self.normalize = normalize
        self.step_pattern = step_pattern

        logger.info(
            f"DTWCalculator инициализирован: "
            f"window={window_size}, metric={distance_metric}, "
            f"normalize={normalize}, pattern={step_pattern}"
        )

    def calculate_dtw(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Расчет DTW дистанции между двумя временными рядами.

        Args:
            series_a: Первый временной ряд
            series_b: Второй временной ряд

        Returns:
            Tuple[float, np.ndarray]: (dtw_distance, cost_matrix)
        """
        # Нормализация рядов
        if self.normalize:
            series_a = self._normalize_series(series_a)
            series_b = self._normalize_series(series_b)

        # Вычисление DTW
        if NUMBA_AVAILABLE and self.window_size is None:
            # Быстрая версия с numba
            dtw_distance, cost_matrix = self._dtw_numba(
                series_a, series_b
            )
        else:
            # Обычная версия (может быть медленнее)
            dtw_distance, cost_matrix = self._dtw_python(
                series_a, series_b
            )

        return dtw_distance, cost_matrix

    def calculate_dtw_distance_normalized(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray
    ) -> float:
        """
        Рассчитывает нормализованную DTW дистанцию в диапазоне [0, 1].

        Args:
            series_a: Первый временной ряд
            series_b: Второй временной ряд

        Returns:
            float: Нормализованная дистанция [0, 1]
        """
        dtw_distance, _ = self.calculate_dtw(series_a, series_b)

        # Нормализация дистанции
        # Максимальная возможная дистанция = sqrt(n * max_point_distance^2)
        n = max(len(series_a), len(series_b))

        if self.normalize:
            # Для нормализованных рядов максимальное расстояние между точками ~2
            max_point_distance = 2.0
        else:
            # Для ненормализованных берем максимальную разницу
            max_point_distance = max(
                np.max(np.abs(series_a)),
                np.max(np.abs(series_b))
            ) * 2

        # Максимальная DTW дистанция при полном несовпадении
        max_dtw = np.sqrt(n * (max_point_distance ** 2))

        # Нормализуем
        normalized = min(dtw_distance / max_dtw, 1.0) if max_dtw > 0 else 0.0

        return float(normalized)

    def _normalize_series(self, series: np.ndarray) -> np.ndarray:
        """Z-score нормализация временного ряда."""
        mean = np.mean(series)
        std = np.std(series)

        if std < 1e-8:
            return np.zeros_like(series)

        return (series - mean) / std

    def _point_distance(self, point_a: float, point_b: float) -> float:
        """Расстояние между двумя точками."""
        if self.distance_metric == 'euclidean':
            return abs(point_a - point_b)
        elif self.distance_metric == 'manhattan':
            return abs(point_a - point_b)
        else:
            return abs(point_a - point_b)

    @staticmethod
    @jit(nopython=True)
    def _dtw_core(series_a, series_b, window_size):
        """
        Ядро DTW алгоритма (оптимизировано с numba).

        Args:
            series_a: Первый ряд
            series_b: Второй ряд
            window_size: Размер окна (или -1 для без ограничений)

        Returns:
            Tuple[float, np.ndarray]: (distance, cost_matrix)
        """
        n, m = len(series_a), len(series_b)

        # Инициализация cost matrix
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, 0] = 0.0

        # Заполнение cost matrix
        for i in range(1, n + 1):
            # Определяем допустимый диапазон для j (Sakoe-Chiba band)
            if window_size > 0:
                j_start = max(1, i - window_size)
                j_end = min(m + 1, i + window_size + 1)
            else:
                j_start = 1
                j_end = m + 1

            for j in range(j_start, j_end):
                # Расстояние между точками
                distance = abs(series_a[i - 1] - series_b[j - 1])

                # Symmetric2 step pattern
                # (i-1, j-1), (i-1, j), (i, j-1)
                cost = min(
                    cost_matrix[i - 1, j - 1],  # Diagonal
                    cost_matrix[i - 1, j],      # Vertical
                    cost_matrix[i, j - 1]       # Horizontal
                )

                cost_matrix[i, j] = distance + cost

        return cost_matrix[n, m], cost_matrix

    def _dtw_python(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Python реализация DTW (используется если numba недоступна).

        Args:
            series_a: Первый ряд
            series_b: Второй ряд

        Returns:
            Tuple[float, np.ndarray]: (distance, cost_matrix)
        """
        n, m = len(series_a), len(series_b)

        # Инициализация cost matrix
        cost_matrix = np.full((n + 1, m + 1), np.inf)
        cost_matrix[0, 0] = 0.0

        # Заполнение cost matrix
        for i in range(1, n + 1):
            # Sakoe-Chiba band
            if self.window_size:
                j_start = max(1, i - self.window_size)
                j_end = min(m + 1, i + self.window_size + 1)
            else:
                j_start = 1
                j_end = m + 1

            for j in range(j_start, j_end):
                # Расстояние между точками
                distance = self._point_distance(
                    series_a[i - 1],
                    series_b[j - 1]
                )

                # Step pattern
                if self.step_pattern == 'symmetric2':
                    # Стандартный symmetric pattern
                    cost = min(
                        cost_matrix[i - 1, j - 1],
                        cost_matrix[i - 1, j],
                        cost_matrix[i, j - 1]
                    )
                elif self.step_pattern == 'symmetric1':
                    # Более строгий - только диагональ + одна сторона
                    cost = min(
                        cost_matrix[i - 1, j - 1],
                        cost_matrix[i - 1, j] + distance,
                        cost_matrix[i, j - 1] + distance
                    )
                elif self.step_pattern == 'asymmetric':
                    # Асимметричный - предпочитает определенное направление
                    cost = min(
                        cost_matrix[i - 1, j - 1],
                        cost_matrix[i - 1, j],
                        cost_matrix[i - 1, j - 2] if j > 1 else np.inf
                    )
                else:
                    cost = cost_matrix[i - 1, j - 1]

                cost_matrix[i, j] = distance + cost

        return cost_matrix[n, m], cost_matrix

    def _dtw_numba(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Быстрая версия DTW с numba.

        Args:
            series_a: Первый ряд
            series_b: Второй ряд

        Returns:
            Tuple[float, np.ndarray]: (distance, cost_matrix)
        """
        window = self.window_size if self.window_size else -1
        return self._dtw_core(series_a, series_b, window)

    def get_warping_path(
        self,
        cost_matrix: np.ndarray
    ) -> list[Tuple[int, int]]:
        """
        Восстанавливает оптимальный warping path из cost matrix.

        Args:
            cost_matrix: Матрица стоимостей из calculate_dtw

        Returns:
            List[Tuple[int, int]]: Список пар индексов (i, j)
        """
        n, m = cost_matrix.shape
        i, j = n - 1, m - 1
        path = [(i - 1, j - 1)]

        while i > 1 or j > 1:
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                # Выбираем минимальную стоимость
                costs = [
                    cost_matrix[i - 1, j - 1],  # Diagonal
                    cost_matrix[i - 1, j],      # Vertical
                    cost_matrix[i, j - 1]       # Horizontal
                ]
                min_idx = np.argmin(costs)

                if min_idx == 0:
                    i -= 1
                    j -= 1
                elif min_idx == 1:
                    i -= 1
                else:
                    j -= 1

            path.append((i - 1, j - 1))

        path.reverse()
        return path

    def calculate_dtw_with_window_optimization(
        self,
        series_a: np.ndarray,
        series_b: np.ndarray,
        max_window_ratio: float = 0.1
    ) -> float:
        """
        Автоматически определяет оптимальный размер окна.

        Args:
            series_a: Первый ряд
            series_b: Второй ряд
            max_window_ratio: Максимальный размер окна как доля от длины

        Returns:
            float: Нормализованная DTW дистанция
        """
        n = max(len(series_a), len(series_b))
        optimal_window = int(n * max_window_ratio)

        # Временно меняем размер окна
        original_window = self.window_size
        self.window_size = optimal_window

        try:
            distance = self.calculate_dtw_distance_normalized(
                series_a, series_b
            )
            return distance
        finally:
            # Восстанавливаем оригинальный размер окна
            self.window_size = original_window


# Глобальный экземпляр для использования в других модулях
dtw_calculator = DTWCalculator()
