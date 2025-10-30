"""
Методы группировки коррелирующих активов.

Путь: backend/strategy/correlation/grouping_methods.py
"""
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from collections import defaultdict

try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from core.logger import get_logger
from .models import AdvancedCorrelationGroup, GroupingMethod

logger = get_logger(__name__)


class GraphBasedGroupManager:
    """
    Группировка на основе графов (Louvain community detection).
    """

    def __init__(self, correlation_threshold: float = 0.7):
        """
        Инициализация.

        Args:
            correlation_threshold: Порог для создания ребер в графе
        """
        self.correlation_threshold = correlation_threshold

        if not NETWORKX_AVAILABLE:
            logger.warning(
                "NetworkX недоступен. "
                "Louvain group вернется к жадной группировке."
            )

    def create_groups_louvain(
        self,
        symbols: List[str],
        correlation_matrix: Dict[Tuple[str, str], float],
        dtw_matrix: Optional[Dict[Tuple[str, str], float]] = None
    ) -> List[AdvancedCorrelationGroup]:
        """
        Создание групп методом Louvain community detection.

        Args:
            symbols: Список символов
            correlation_matrix: Матрица корреляций
            dtw_matrix: Матрица DTW (опционально)

        Returns:
            List[AdvancedCorrelationGroup]: Список групп
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX недоступен, используется fallback")
            return self._create_groups_greedy_fallback(
                symbols, correlation_matrix
            )

        logger.info("Группировка методом Louvain...")

        # Создаем граф
        G = nx.Graph()

        # Добавляем вершины
        G.add_nodes_from(symbols)

        # Добавляем ребра (только для сильно коррелирующих пар)
        edge_count = 0
        for (sym1, sym2), corr in correlation_matrix.items():
            if abs(corr) >= self.correlation_threshold:
                # Вес ребра = корреляция
                G.add_edge(sym1, sym2, weight=abs(corr))
                edge_count += 1

        logger.info(f"Создан граф: {len(symbols)} вершин, {edge_count} ребер")

        # Louvain community detection
        try:
            communities = community.greedy_modularity_communities(
                G, weight='weight'
            )

            groups = []
            for idx, comm in enumerate(communities):
                symbols_in_group = sorted(list(comm))

                if len(symbols_in_group) < 2:
                    # Одиночные символы игнорируем
                    continue

                # Рассчитываем метрики группы
                metrics = self._calculate_group_metrics(
                    symbols_in_group,
                    correlation_matrix,
                    dtw_matrix
                )

                group = AdvancedCorrelationGroup(
                    group_id=f"louvain_{idx}",
                    symbols=symbols_in_group,
                    grouping_method=GroupingMethod.LOUVAIN,
                    avg_correlation=metrics['avg_correlation'],
                    min_correlation=metrics['min_correlation'],
                    max_correlation=metrics['max_correlation'],
                    avg_dtw_distance=metrics.get('avg_dtw', 0.0),
                    avg_volatility=0.0,  # TODO: добавить расчет
                    cluster_quality_score=0.0  # TODO: добавить расчет
                )

                groups.append(group)

            logger.info(f"Создано {len(groups)} групп методом Louvain")
            return groups

        except Exception as e:
            logger.error(f"Ошибка Louvain: {e}", exc_info=True)
            return self._create_groups_greedy_fallback(symbols, correlation_matrix)

    def _create_groups_greedy_fallback(
        self,
        symbols: List[str],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> List[AdvancedCorrelationGroup]:
        """Fallback к жадной группировке."""
        logger.info("Используется жадная группировка (fallback)")

        groups = []
        processed: Set[str] = set()
        group_counter = 0

        for symbol in symbols:
            if symbol in processed:
                continue

            # Создаем новую группу
            group_symbols = {symbol}

            # Ищем коррелирующие символы
            for other_symbol in symbols:
                if other_symbol == symbol or other_symbol in processed:
                    continue

                key = (
                    (symbol, other_symbol)
                    if symbol < other_symbol
                    else (other_symbol, symbol)
                )
                correlation = correlation_matrix.get(key, 0.0)

                if abs(correlation) >= self.correlation_threshold:
                    group_symbols.add(other_symbol)

            if len(group_symbols) > 1:
                metrics = self._calculate_group_metrics(
                    list(group_symbols),
                    correlation_matrix,
                    None
                )

                group = AdvancedCorrelationGroup(
                    group_id=f"greedy_{group_counter}",
                    symbols=sorted(list(group_symbols)),
                    grouping_method=GroupingMethod.GREEDY,
                    avg_correlation=metrics['avg_correlation'],
                    min_correlation=metrics['min_correlation'],
                    max_correlation=metrics['max_correlation'],
                    avg_dtw_distance=0.0,
                    avg_volatility=0.0,
                    cluster_quality_score=0.0
                )

                groups.append(group)

                for s in group_symbols:
                    processed.add(s)

                group_counter += 1
            else:
                processed.add(symbol)

        return groups

    def _calculate_group_metrics(
        self,
        symbols: List[str],
        correlation_matrix: Dict[Tuple[str, str], float],
        dtw_matrix: Optional[Dict[Tuple[str, str], float]]
    ) -> Dict:
        """Расчет метрик группы."""
        correlations = []
        dtw_distances = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                key = (sym1, sym2) if sym1 < sym2 else (sym2, sym1)

                if key in correlation_matrix:
                    correlations.append(abs(correlation_matrix[key]))

                if dtw_matrix and key in dtw_matrix:
                    dtw_distances.append(dtw_matrix[key])

        return {
            'avg_correlation': np.mean(correlations) if correlations else 0.0,
            'min_correlation': np.min(correlations) if correlations else 0.0,
            'max_correlation': np.max(correlations) if correlations else 0.0,
            'avg_dtw': np.mean(dtw_distances) if dtw_distances else 0.0
        }


class HierarchicalGroupManager:
    """
    Группировка методом hierarchical clustering.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.7,
        linkage: str = 'ward'
    ):
        """
        Инициализация.

        Args:
            correlation_threshold: Порог для определения количества кластеров
            linkage: Метод linkage (ward, average, complete)
        """
        self.correlation_threshold = correlation_threshold
        self.linkage = linkage

        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn недоступен. Hierarchical clustering недоступен.")

    def create_groups_hierarchical(
        self,
        symbols: List[str],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> List[AdvancedCorrelationGroup]:
        """
        Создание групп методом hierarchical clustering.

        Args:
            symbols: Список символов
            correlation_matrix: Матрица корреляций

        Returns:
            List[AdvancedCorrelationGroup]: Список групп
        """
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn недоступен")
            return []

        logger.info("Группировка методом Hierarchical Clustering...")

        # Преобразуем correlation matrix в distance matrix
        distance_matrix = self._correlation_to_distance_matrix(
            symbols, correlation_matrix
        )

        try:
            # Определяем оптимальное количество кластеров
            n_clusters = self._estimate_n_clusters(len(symbols))

            # Hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage=self.linkage
            )

            labels = clustering.fit_predict(distance_matrix)

            # Группируем символы по кластерам
            clusters = defaultdict(list)
            for symbol, label in zip(symbols, labels):
                clusters[label].append(symbol)

            # Создаем группы
            groups = []
            for idx, symbols_in_cluster in clusters.items():
                if len(symbols_in_cluster) < 2:
                    continue

                metrics = self._calculate_group_metrics(
                    symbols_in_cluster,
                    correlation_matrix
                )

                group = AdvancedCorrelationGroup(
                    group_id=f"hierarchical_{idx}",
                    symbols=sorted(symbols_in_cluster),
                    grouping_method=GroupingMethod.HIERARCHICAL,
                    avg_correlation=metrics['avg_correlation'],
                    min_correlation=metrics['min_correlation'],
                    max_correlation=metrics['max_correlation'],
                    avg_dtw_distance=0.0,
                    avg_volatility=0.0,
                    cluster_quality_score=0.0
                )

                groups.append(group)

            logger.info(f"Создано {len(groups)} групп методом Hierarchical")
            return groups

        except Exception as e:
            logger.error(f"Ошибка Hierarchical clustering: {e}", exc_info=True)
            return []

    def _correlation_to_distance_matrix(
        self,
        symbols: List[str],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> np.ndarray:
        """Преобразование correlation matrix в distance matrix."""
        n = len(symbols)
        distance_matrix = np.zeros((n, n))

        symbol_to_idx = {sym: idx for idx, sym in enumerate(symbols)}

        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i == j:
                    distance_matrix[i, j] = 0.0
                    continue

                key = (sym1, sym2) if sym1 < sym2 else (sym2, sym1)
                correlation = correlation_matrix.get(key, 0.0)

                # Distance = 1 - correlation (для положительных корреляций)
                # Для [-1, 1] лучше использовать: (1 - correlation) / 2
                distance = (1.0 - correlation) / 2.0
                distance_matrix[i, j] = distance

        return distance_matrix

    def _estimate_n_clusters(self, n_symbols: int) -> int:
        """Оценка оптимального количества кластеров."""
        # Эвристика: примерно sqrt(n/2)
        estimated = max(2, int(np.sqrt(n_symbols / 2)))
        return min(estimated, n_symbols // 2)

    def _calculate_group_metrics(
        self,
        symbols: List[str],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> Dict:
        """Расчет метрик группы."""
        correlations = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                key = (sym1, sym2) if sym1 < sym2 else (sym2, sym1)

                if key in correlation_matrix:
                    correlations.append(abs(correlation_matrix[key]))

        return {
            'avg_correlation': np.mean(correlations) if correlations else 0.0,
            'min_correlation': np.min(correlations) if correlations else 0.0,
            'max_correlation': np.max(correlations) if correlations else 0.0
        }


class EnsembleGroupManager:
    """
    Комбинированная группировка (консенсус нескольких методов).
    """

    def __init__(self, correlation_threshold: float = 0.7):
        """Инициализация."""
        self.correlation_threshold = correlation_threshold
        self.louvain_manager = GraphBasedGroupManager(correlation_threshold)
        self.hierarchical_manager = HierarchicalGroupManager(correlation_threshold)

    def create_groups_ensemble(
        self,
        symbols: List[str],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> List[AdvancedCorrelationGroup]:
        """
        Создание групп методом ensemble (консенсус).

        Args:
            symbols: Список символов
            correlation_matrix: Матрица корреляций

        Returns:
            List[AdvancedCorrelationGroup]: Консенсусные группы
        """
        logger.info("Группировка методом Ensemble...")

        # Получаем группы от разных методов
        louvain_groups = self.louvain_manager.create_groups_louvain(
            symbols, correlation_matrix
        )
        hierarchical_groups = self.hierarchical_manager.create_groups_hierarchical(
            symbols, correlation_matrix
        )

        # Преобразуем в словари symbol -> group_id
        louvain_mapping = self._groups_to_mapping(louvain_groups)
        hierarchical_mapping = self._groups_to_mapping(hierarchical_groups)

        # Находим консенсус (пары, которые в одной группе в обоих методах)
        consensus_pairs = self._find_consensus_pairs(
            louvain_mapping,
            hierarchical_mapping
        )

        # Создаем финальные группы из консенсусных пар
        final_groups = self._build_groups_from_pairs(
            consensus_pairs,
            correlation_matrix
        )

        logger.info(f"Создано {len(final_groups)} консенсусных групп")
        return final_groups

    def _groups_to_mapping(
        self,
        groups: List[AdvancedCorrelationGroup]
    ) -> Dict[str, str]:
        """Преобразование списка групп в mapping symbol -> group_id."""
        mapping = {}
        for group in groups:
            for symbol in group.symbols:
                mapping[symbol] = group.group_id
        return mapping

    def _find_consensus_pairs(
        self,
        mapping1: Dict[str, str],
        mapping2: Dict[str, str]
    ) -> Set[Tuple[str, str]]:
        """Находит пары символов, которые в одной группе в обоих методах."""
        consensus_pairs = set()

        all_symbols = set(mapping1.keys()) | set(mapping2.keys())

        for sym1 in all_symbols:
            for sym2 in all_symbols:
                if sym1 >= sym2:
                    continue

                # Проверяем, в одной ли группе в обоих методах
                in_same_group_1 = (
                    mapping1.get(sym1) == mapping1.get(sym2)
                    and mapping1.get(sym1) is not None
                )
                in_same_group_2 = (
                    mapping2.get(sym1) == mapping2.get(sym2)
                    and mapping2.get(sym1) is not None
                )

                if in_same_group_1 and in_same_group_2:
                    consensus_pairs.add((sym1, sym2))

        return consensus_pairs

    def _build_groups_from_pairs(
        self,
        pairs: Set[Tuple[str, str]],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> List[AdvancedCorrelationGroup]:
        """Строит группы из консенсусных пар."""
        # Используем Union-Find для объединения пар в группы
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Объединяем пары
        for sym1, sym2 in pairs:
            union(sym1, sym2)

        # Группируем по корню
        groups_dict = defaultdict(list)
        for sym in parent.keys():
            root = find(sym)
            groups_dict[root].append(sym)

        # Создаем группы
        final_groups = []
        for idx, symbols in enumerate(groups_dict.values()):
            if len(symbols) < 2:
                continue

            metrics = self._calculate_group_metrics(symbols, correlation_matrix)

            group = AdvancedCorrelationGroup(
                group_id=f"ensemble_{idx}",
                symbols=sorted(symbols),
                grouping_method=GroupingMethod.ENSEMBLE,
                avg_correlation=metrics['avg_correlation'],
                min_correlation=metrics['min_correlation'],
                max_correlation=metrics['max_correlation'],
                avg_dtw_distance=0.0,
                avg_volatility=0.0,
                cluster_quality_score=0.0
            )

            final_groups.append(group)

        return final_groups

    def _calculate_group_metrics(
        self,
        symbols: List[str],
        correlation_matrix: Dict[Tuple[str, str], float]
    ) -> Dict:
        """Расчет метрик группы."""
        correlations = []

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                key = (sym1, sym2) if sym1 < sym2 else (sym2, sym1)

                if key in correlation_matrix:
                    correlations.append(abs(correlation_matrix[key]))

        return {
            'avg_correlation': np.mean(correlations) if correlations else 0.0,
            'min_correlation': np.min(correlations) if correlations else 0.0,
            'max_correlation': np.max(correlations) if correlations else 0.0
        }
