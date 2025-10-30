"""
Модели данных для продвинутого анализа корреляций.

Путь: backend/strategy/correlation/models.py
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from datetime import datetime
from enum import Enum


class CorrelationMethod(str, Enum):
    """Методы расчета корреляции."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    DTW = "dtw"  # Dynamic Time Warping


class GroupingMethod(str, Enum):
    """Методы группировки."""
    GREEDY = "greedy"  # Текущий жадный алгоритм
    LOUVAIN = "louvain"  # Community detection
    HIERARCHICAL = "hierarchical"  # Hierarchical clustering
    KMEANS = "kmeans"  # K-means clustering
    ENSEMBLE = "ensemble"  # Консенсус нескольких методов


class MarketCorrelationRegime(str, Enum):
    """Режимы корреляций на рынке."""
    LOW_CORRELATION = "low_correlation"  # Корреляции низкие (диверсификация работает)
    MODERATE_CORRELATION = "moderate_correlation"  # Умеренные корреляции
    HIGH_CORRELATION = "high_correlation"  # Высокие корреляции (риск концентрации)
    CRISIS_CORRELATION = "crisis_correlation"  # Все падает вместе


@dataclass
class CorrelationMetrics:
    """
    Набор метрик корреляции между двумя символами.

    Attributes:
        symbol_a: Первый символ
        symbol_b: Второй символ
        pearson: Pearson correlation coefficient
        spearman: Spearman rank correlation
        pearson_7d: Pearson за 7 дней
        pearson_14d: Pearson за 14 дней
        pearson_30d: Pearson за 30 дней
        dtw_distance: Dynamic Time Warping distance (нормализовано 0-1)
        volatility_distance: Разница в волатильности (0-1)
        return_sign_agreement: Согласие направления движения (0-1)
        weighted_score: Взвешенная финальная оценка корреляции
        calculated_at: Время расчета
    """
    symbol_a: str
    symbol_b: str
    pearson: float
    spearman: float
    pearson_7d: float
    pearson_14d: float
    pearson_30d: float
    dtw_distance: float
    volatility_distance: float
    return_sign_agreement: float
    weighted_score: float
    calculated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Валидация метрик."""
        assert -1.0 <= self.pearson <= 1.0, "Pearson должен быть в [-1, 1]"
        assert -1.0 <= self.spearman <= 1.0, "Spearman должен быть в [-1, 1]"
        assert 0.0 <= self.dtw_distance <= 1.0, "DTW distance должен быть в [0, 1]"
        assert 0.0 <= self.return_sign_agreement <= 1.0, "Sign agreement должен быть в [0, 1]"


@dataclass
class RollingCorrelationWindow:
    """
    Параметры rolling window для расчета корреляции.

    Attributes:
        window_days: Размер окна в днях
        weight: Вес этого окна в финальном расчете
        correlation: Рассчитанная корреляция для этого окна
    """
    window_days: int
    weight: float
    correlation: Optional[float] = None

    def __post_init__(self):
        """Валидация."""
        assert self.window_days > 0, "Window должен быть положительным"
        assert 0.0 <= self.weight <= 1.0, "Weight должен быть в [0, 1]"


@dataclass
class AdvancedCorrelationGroup:
    """
    Продвинутая группа коррелирующих символов.

    Attributes:
        group_id: Уникальный ID группы
        symbols: Список символов в группе
        grouping_method: Метод, которым создана группа
        avg_correlation: Средняя корреляция между символами
        min_correlation: Минимальная корреляция в группе
        max_correlation: Максимальная корреляция в группе
        avg_dtw_distance: Средняя DTW дистанция
        avg_volatility: Средняя волатильность группы
        cluster_quality_score: Качество кластера (silhouette score)
        active_positions: Количество открытых позиций
        total_exposure_usdt: Общая экспозиция группы в USDT
        created_at: Время создания группы
        last_updated: Время последнего обновления
    """
    group_id: str
    symbols: List[str]
    grouping_method: GroupingMethod
    avg_correlation: float
    min_correlation: float
    max_correlation: float
    avg_dtw_distance: float
    avg_volatility: float
    cluster_quality_score: float
    active_positions: int = 0
    total_exposure_usdt: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class VolatilityCluster:
    """
    Кластер активов по волатильности.

    Attributes:
        cluster_id: ID кластера
        volatility_range: Диапазон волатильности (min, max)
        symbols: Символы в кластере
        avg_volatility: Средняя волатильность кластера
    """
    cluster_id: str
    volatility_range: tuple[float, float]
    symbols: List[str]
    avg_volatility: float


@dataclass
class CorrelationRegimeInfo:
    """
    Информация о текущем режиме корреляций на рынке.

    Attributes:
        regime: Тип режима
        avg_market_correlation: Средняя корреляция по рынку
        correlation_threshold: Рекомендуемый порог группировки
        max_positions_per_group: Рекомендуемый лимит позиций
        high_correlation_pairs_count: Количество сильно коррелирующих пар
        independent_pairs_count: Количество независимых пар
        regime_confidence: Уверенность в определении режима
        detected_at: Время определения режима
    """
    regime: MarketCorrelationRegime
    avg_market_correlation: float
    correlation_threshold: float
    max_positions_per_group: int
    high_correlation_pairs_count: int
    independent_pairs_count: int
    regime_confidence: float
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class GroupingComparisonResult:
    """
    Результат сравнения разных методов группировки (для ensemble).

    Attributes:
        louvain_groups: Группы из Louvain
        hierarchical_groups: Группы из Hierarchical
        kmeans_groups: Группы из K-means
        consensus_groups: Консенсусные группы
        agreement_score: Степень согласия методов (0-1)
    """
    louvain_groups: Dict[str, List[str]]
    hierarchical_groups: Dict[str, List[str]]
    kmeans_groups: Dict[str, List[str]]
    consensus_groups: Dict[str, List[str]]
    agreement_score: float


@dataclass
class DTWParameters:
    """
    Параметры для Dynamic Time Warping.

    Attributes:
        max_lag_hours: Максимальный лаг в часах
        window_size_hours: Размер окна в часах
        distance_measure: Метрика расстояния
        normalize: Нормализовать ли временные ряды
    """
    max_lag_hours: int = 24
    window_size_hours: int = 168  # 7 дней
    distance_measure: Literal["euclidean", "manhattan"] = "euclidean"
    normalize: bool = True


@dataclass
class ConditionalCorrelationMetrics:
    """
    Корреляции в различных рыночных условиях.

    Attributes:
        symbol_a: Первый символ
        symbol_b: Второй символ
        bullish_correlation: Корреляция в бычьем тренде
        bearish_correlation: Корреляция в медвежьем тренде
        high_vol_correlation: Корреляция при высокой волатильности
        low_vol_correlation: Корреляция при низкой волатильности
        crisis_correlation: Корреляция во время кризисов
    """
    symbol_a: str
    symbol_b: str
    bullish_correlation: Optional[float] = None
    bearish_correlation: Optional[float] = None
    high_vol_correlation: Optional[float] = None
    low_vol_correlation: Optional[float] = None
    crisis_correlation: Optional[float] = None
