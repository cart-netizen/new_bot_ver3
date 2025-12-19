"""
Feature Store Schema - определяет структуру данных Feature Store

Определяет 112 фич для модели HybridCNNLSTM:
- OrderBook features: 50
- Candle features: 25
- Indicator features: 37
"""

from typing import List, Set
from dataclasses import dataclass, field
import pandas as pd

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureStoreSchema:
    """
    Схема данных Feature Store

    Определяет все колонки и их типы для обучения модели.
    Гарантирует консистентность между Feature Store и моделью.
    """

    # ===== META COLUMNS =====
    timestamp_column: str = 'timestamp'
    symbol_column: str = 'symbol'

    # ===== LABEL COLUMNS =====
    # Changed to 300s (5 min) horizon for better predictability
    label_column: str = 'future_direction_300s'  # 0=DOWN, 1=NEUTRAL, 2=UP
    return_column: str = 'future_movement_300s'  # Price movement percentage

    # Alternative horizons (for multi-horizon training)
    label_column_60s: str = 'future_direction_60s'
    label_column_180s: str = 'future_direction_180s'
    label_column_300s: str = 'future_direction_300s'

    # ===== FEATURE COLUMNS =====
    # Будут генерироваться в __post_init__
    orderbook_features: List[str] = field(default_factory=list)
    candle_features: List[str] = field(default_factory=list)
    indicator_features: List[str] = field(default_factory=list)

    # ===== EXTENDED FEATURES (V2) =====
    # Lagged и derived фичи добавляются preprocessing_v2
    lagged_features: List[str] = field(default_factory=list)
    derived_features: List[str] = field(default_factory=list)

    # Флаг использования расширенных фич
    use_extended_features: bool = True

    def __post_init__(self):
        """Generate feature lists if not provided"""
        if not self.orderbook_features:
            self.orderbook_features = self._generate_orderbook_features()

        if not self.candle_features:
            self.candle_features = self._generate_candle_features()

        if not self.indicator_features:
            self.indicator_features = self._generate_indicator_features()

        if not self.lagged_features:
            self.lagged_features = self._generate_lagged_features()

        if not self.derived_features:
            self.derived_features = self._generate_derived_features()

        # Log feature counts
        base_total = len(self.orderbook_features) + len(self.candle_features) + len(self.indicator_features)
        extended_total = len(self.lagged_features) + len(self.derived_features)

        logger.info(
            f"Feature schema: base={base_total}, extended={extended_total}, "
            f"total={base_total + extended_total}"
        )

    def _generate_orderbook_features(self) -> List[str]:
        """
        Generate OrderBook feature names (50 features)

        Structure:
        - Базовые микроструктурные: 15
        - Дисбаланс и давление: 10
        - Кластеры и уровни: 10
        - Ликвидность: 8
        - Временные: 7
        """
        features = [
            # Базовые микроструктурные (15)
            'bid_ask_spread_abs',
            'bid_ask_spread_rel',
            'mid_price',
            'micro_price',
            'vwap_bid_5',
            'vwap_ask_5',
            'vwap_bid_10',
            'vwap_ask_10',
            'depth_bid_5',
            'depth_ask_5',
            'depth_bid_10',
            'depth_ask_10',
            'total_bid_volume',
            'total_ask_volume',
            'book_depth_ratio',

            # Дисбаланс и давление (10)
            'imbalance_5',
            'imbalance_10',
            'imbalance_total',
            'price_pressure',
            'volume_delta_5',
            'order_flow_imbalance',
            'bid_intensity',
            'ask_intensity',
            'buy_sell_ratio',
            'smart_money_index',

            # Кластеры и уровни (10)
            'largest_bid_cluster_price',
            'largest_bid_cluster_volume',
            'largest_ask_cluster_price',
            'largest_ask_cluster_volume',
            'num_bid_clusters',
            'num_ask_clusters',
            'support_level_1',
            'resistance_level_1',
            'distance_to_support',
            'distance_to_resistance',

            # Ликвидность (8)
            'liquidity_bid_5',
            'liquidity_ask_5',
            'liquidity_asymmetry',
            'effective_spread',
            'kyle_lambda',
            'amihud_illiquidity',
            'roll_spread',
            'depth_imbalance_ratio',

            # Временные (7)
            'level_ttl_avg',
            'level_ttl_std',
            'orderbook_volatility',
            'update_frequency',
            'quote_intensity',
            'trade_arrival_rate',
            'spread_volatility'
        ]

        assert len(features) == 50, f"OrderBook features: expected 50, got {len(features)}"
        return features

    def _generate_candle_features(self) -> List[str]:
        """
        Generate Candle feature names (25 features)

        Structure:
        - Базовые OHLCV: 6
        - Производные метрики: 7
        - Волатильность: 3
        - Volume features: 5
        - Pattern indicators: 4
        """
        features = [
            # Базовые OHLCV (6)
            'open',
            'high',
            'low',
            'close',
            'volume',
            'typical_price',

            # Производные метрики (7)
            'returns',
            'log_returns',
            'high_low_range',
            'close_open_diff',
            'upper_shadow',
            'lower_shadow',
            'body_size',

            # Волатильность (3)
            'realized_volatility',
            'parkinson_volatility',
            'garman_klass_volatility',

            # Volume features (5)
            'volume_ma_ratio',
            'volume_change_rate',
            'price_volume_trend',
            'volume_weighted_price',
            'money_flow',

            # Pattern indicators (4)
            'doji_strength',
            'hammer_strength',
            'engulfing_strength',
            'gap_size'
        ]

        assert len(features) == 25, f"Candle features: expected 25, got {len(features)}"
        return features

    def _generate_indicator_features(self) -> List[str]:
        """
        Generate Indicator feature names (37 features)

        Structure:
        - Trend indicators: 12
        - Momentum indicators: 9
        - Volatility indicators: 8
        - Volume indicators: 6
        - Aroon indicators: 2
        """
        features = [
            # Trend indicators (12)
            'sma_10',
            'sma_20',
            'sma_50',
            'ema_10',
            'ema_20',
            'ema_50',
            'macd',
            'macd_signal',
            'macd_histogram',
            'adx',
            'plus_di',
            'minus_di',

            # Momentum indicators (9)
            'rsi_14',
            'rsi_28',
            'stochastic_k',
            'stochastic_d',
            'williams_r',
            'cci',
            'momentum_10',
            'roc',
            'mfi',

            # Volatility indicators (8)
            'bollinger_upper',
            'bollinger_middle',
            'bollinger_lower',
            'bollinger_width',
            'bollinger_pct',
            'atr_14',
            'keltner_upper',
            'keltner_lower',

            # Volume indicators (6)
            'obv',
            'vwap',
            'ad_line',
            'cmf',
            'vpt',
            'nvi',

            # Aroon indicators (2)
            'aroon_up',
            'aroon_down'
        ]

        assert len(features) == 37, f"Indicator features: expected 37, got {len(features)}"
        return features

    def _generate_lagged_features(self) -> List[str]:
        """
        Generate lagged feature names.

        Лаги для top-коррелированных и top-сепарирующих фич.
        Каждая фича получает лаги: 1, 2, 4, 8 (при 15s интервале: 15s, 30s, 1m, 2m)

        Total: 15 base features × 4 lags = 60 lagged features
        """
        # Top-корреляция + top-сепарабельность фичи
        base_features = [
            'imbalance_5',
            'depth_imbalance_ratio',
            'imbalance_10',
            'volume_delta_5',
            'gap_size',
            'momentum_10',
            'update_frequency',
            'quote_intensity',
            'rsi_28',
            'roc',
            'orderbook_volatility',
            'bid_ask_spread_rel',
            'effective_spread',
            'trade_arrival_rate',
            'smart_money_index',
        ]

        lags = [1, 2, 4, 8]
        features = []

        for feat in base_features:
            for lag in lags:
                features.append(f"{feat}_lag{lag}")

        return features

    def _generate_derived_features(self) -> List[str]:
        """
        Generate derived feature names.

        Производные фичи на основе top-корреляций:
        - Momentum/change фичи
        - Ratio фичи
        - Volatility-adjusted фичи
        - Composite signals

        Total: ~15 derived features
        """
        return [
            # Imbalance derivatives
            'imbalance_5_change',
            'imbalance_5_change_pct',
            'imbalance_5_momentum',
            'imbalance_ratio_5_10',
            'imbalance_vol_adjusted',
            'imbalance_volume_weighted',

            # Spread derivatives
            'spread_change',
            'spread_momentum',

            # RSI derivatives
            'rsi_diff',
            'rsi_14_momentum',

            # Composite signals
            'composite_signal',

            # Orderbook pressure
            'smart_money_momentum',
            'smart_money_acceleration',
            'trade_quote_ratio',
            'volatility_regime',
        ]

    def get_all_feature_columns(self, include_extended: bool = None) -> List[str]:
        """
        Get list of all feature columns.

        Args:
            include_extended: Include lagged/derived features.
                              If None, uses self.use_extended_features

        Returns:
            List of all feature column names in order
        """
        base_features = (
            self.orderbook_features +
            self.candle_features +
            self.indicator_features
        )

        if include_extended is None:
            include_extended = self.use_extended_features

        if include_extended:
            return base_features + self.lagged_features + self.derived_features
        else:
            return base_features

    def get_base_feature_columns(self) -> List[str]:
        """Get only base 112 features (without lagged/derived)."""
        return (
            self.orderbook_features +
            self.candle_features +
            self.indicator_features
        )

    def get_required_columns(self) -> List[str]:
        """
        Get all required columns (features + meta + labels)

        Returns:
            List of all required column names
        """
        return (
            [self.timestamp_column, self.symbol_column] +
            self.get_all_feature_columns() +
            [self.label_column]
        )

    def get_optional_columns(self) -> List[str]:
        """
        Get optional columns

        Returns:
            List of optional column names
        """
        return [
            self.return_column,
            # Alternative horizons
            'future_direction_60s',
            'future_movement_60s',
            'future_direction_180s',
            'future_movement_180s',
            # Legacy columns
            'future_direction_10s',
            'future_movement_10s',
            'future_direction_30s',
            'future_movement_30s',
            # Current mid price (used in preprocessing)
            'current_mid_price',
        ]

    def validate_dataframe(self, df: pd.DataFrame, strict: bool = True) -> bool:
        """
        Validate that DataFrame has all required columns

        Args:
            df: DataFrame to validate
            strict: If True, raise error on missing columns.
                   If False, only log warning.

        Returns:
            True if valid

        Raises:
            ValueError: If strict=True and validation fails
        """
        required = set(self.get_required_columns())
        available = set(df.columns)

        missing = required - available
        extra = available - required - set(self.get_optional_columns())

        if missing:
            msg = f"Missing required columns: {sorted(missing)}"
            logger.error(msg)
            if strict:
                raise ValueError(msg)
            return False

        if extra:
            logger.info(f"Extra columns found (will be ignored): {sorted(extra)[:10]}...")

        logger.info(
            f"✓ DataFrame validation passed: "
            f"{len(self.get_all_feature_columns())} features, "
            f"{len(df)} rows"
        )
        return True

    def filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to keep only required columns

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with only required columns
        """
        required = self.get_required_columns()
        optional = self.get_optional_columns()

        # Keep required + available optional
        keep_cols = [c for c in required if c in df.columns]
        keep_cols += [c for c in optional if c in df.columns]

        if len(keep_cols) < len(required):
            missing = set(required) - set(keep_cols)
            raise ValueError(f"Cannot filter: missing required columns {missing}")

        return df[keep_cols]

    def get_feature_groups(self, include_extended: bool = True) -> dict:
        """
        Get feature groups for analysis

        Args:
            include_extended: Include lagged/derived feature groups

        Returns:
            Dict mapping group name to feature list
        """
        groups = {
            'orderbook': self.orderbook_features,
            'candle': self.candle_features,
            'indicator': self.indicator_features,
        }

        if include_extended:
            groups['lagged'] = self.lagged_features
            groups['derived'] = self.derived_features

        return groups

    def summary(self) -> str:
        """
        Get summary of schema

        Returns:
            Formatted summary string
        """
        base_count = len(self.get_base_feature_columns())
        extended_count = len(self.lagged_features) + len(self.derived_features)
        total_count = base_count + extended_count

        lines = [
            "Feature Store Schema Summary (V2)",
            "=" * 60,
            f"Base features: {base_count}",
            f"  • OrderBook: {len(self.orderbook_features)}",
            f"  • Candle: {len(self.candle_features)}",
            f"  • Indicator: {len(self.indicator_features)}",
            "",
            f"Extended features: {extended_count}",
            f"  • Lagged: {len(self.lagged_features)}",
            f"  • Derived: {len(self.derived_features)}",
            "",
            f"TOTAL FEATURES: {total_count}",
            "",
            f"Meta columns: {self.timestamp_column}, {self.symbol_column}",
            f"Label column: {self.label_column} (5 min horizon)",
            f"Return column: {self.return_column}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ===================================================================
# Global Default Schema
# ===================================================================

DEFAULT_SCHEMA = FeatureStoreSchema()

logger.info("✓ FeatureStoreSchema initialized with 112 features")
logger.debug(DEFAULT_SCHEMA.summary())
