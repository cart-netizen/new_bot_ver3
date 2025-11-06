"""
Feature Store Schema - определяет структуру данных Feature Store

Определяет 110 фич для модели HybridCNNLSTM:
- OrderBook features: 50
- Candle features: 25
- Indicator features: 35
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
    label_column: str = 'future_direction_60s'  # 0=DOWN, 1=NEUTRAL, 2=UP
    return_column: str = 'future_return_60s'     # Expected return

    # ===== FEATURE COLUMNS =====
    # Будут генерироваться в __post_init__
    orderbook_features: List[str] = field(default_factory=list)
    candle_features: List[str] = field(default_factory=list)
    indicator_features: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Generate feature lists if not provided"""
        if not self.orderbook_features:
            self.orderbook_features = self._generate_orderbook_features()

        if not self.candle_features:
            self.candle_features = self._generate_candle_features()

        if not self.indicator_features:
            self.indicator_features = self._generate_indicator_features()

        # Validate total count
        total = len(self.orderbook_features) + len(self.candle_features) + len(self.indicator_features)
        if total != 110:
            logger.warning(
                f"Feature count mismatch: expected 110, got {total} "
                f"(orderbook={len(self.orderbook_features)}, "
                f"candle={len(self.candle_features)}, "
                f"indicator={len(self.indicator_features)})"
            )

    def _generate_orderbook_features(self) -> List[str]:
        """
        Generate OrderBook feature names (50 features)

        Structure:
        - 10 levels × 4 metrics = 40 features
        - Aggregated metrics = 10 features
        """
        features = []

        # Level-by-level features (10 levels × 4 = 40)
        for i in range(10):
            features.extend([
                f'bid_price_level_{i}',
                f'bid_volume_level_{i}',
                f'ask_price_level_{i}',
                f'ask_volume_level_{i}',
            ])

        # Aggregated orderbook features (10)
        features.extend([
            'orderbook_imbalance',        # (bid_vol - ask_vol) / (bid_vol + ask_vol)
            'weighted_mid_price',          # Volume-weighted mid price
            'total_bid_volume',            # Sum of all bid volumes
            'total_ask_volume',            # Sum of all ask volumes
            'bid_ask_spread',              # Absolute spread
            'spread_bps',                  # Spread in basis points
            'vwap_distance',               # Distance from VWAP
            'depth_imbalance_5',           # Imbalance for top 5 levels
            'depth_imbalance_10',          # Imbalance for all 10 levels
            'microprice',                  # Microprice indicator
        ])

        assert len(features) == 50, f"OrderBook features: expected 50, got {len(features)}"
        return features

    def _generate_candle_features(self) -> List[str]:
        """
        Generate Candle feature names (25 features)

        Structure:
        - OHLCV + derived = 10 features
        - Moving averages = 6 features
        - Returns & volatility = 6 features
        - Other = 3 features
        """
        features = [
            # Basic OHLCV (5)
            'open',
            'high',
            'low',
            'close',
            'volume',

            # Derived from OHLC (5)
            'hl_range',           # high - low
            'oc_range',           # open - close
            'upper_shadow',       # high - max(open, close)
            'lower_shadow',       # min(open, close) - low
            'body_size',          # abs(close - open)

            # Moving averages (6)
            'close_ma_5',         # 5-period MA
            'close_ma_20',        # 20-period MA
            'close_ma_50',        # 50-period MA
            'volume_ma_5',
            'volume_ma_20',
            'volume_ma_50',

            # Returns (3)
            'return_1m',          # 1-minute return
            'return_5m',          # 5-minute return
            'return_15m',         # 15-minute return

            # Volatility (3)
            'volatility_1h',      # 1-hour rolling std
            'volatility_4h',      # 4-hour rolling std
            'volatility_24h',     # 24-hour rolling std

            # Other (3)
            'vwap',               # Volume-weighted average price
            'typical_price',      # (high + low + close) / 3
            'money_flow',         # typical_price * volume
        ]

        assert len(features) == 25, f"Candle features: expected 25, got {len(features)}"
        return features

    def _generate_indicator_features(self) -> List[str]:
        """
        Generate Indicator feature names (35 features)

        Structure:
        - RSI variants = 3
        - MACD variants = 3
        - Bollinger Bands = 4
        - Volume indicators = 5
        - Momentum indicators = 8
        - Trend indicators = 7
        - Other = 5
        """
        features = [
            # RSI variants (3)
            'rsi_14',             # Standard RSI(14)
            'rsi_7',              # Fast RSI(7)
            'rsi_28',             # Slow RSI(28)

            # MACD (3)
            'macd',               # MACD line
            'macd_signal',        # Signal line
            'macd_histogram',     # Histogram

            # Bollinger Bands (4)
            'bb_upper',           # Upper band
            'bb_middle',          # Middle band (SMA)
            'bb_lower',           # Lower band
            'bb_width',           # Band width

            # Volume indicators (5)
            'obv',                # On-Balance Volume
            'volume_roc',         # Volume Rate of Change
            'vpt',                # Volume Price Trend
            'mfi',                # Money Flow Index
            'adl',                # Accumulation/Distribution Line

            # Momentum indicators (8)
            'momentum',           # Momentum
            'roc',                # Rate of Change
            'stoch_k',            # Stochastic %K
            'stoch_d',            # Stochastic %D
            'williams_r',         # Williams %R
            'cci',                # Commodity Channel Index
            'ultimate_oscillator', # Ultimate Oscillator
            'awesome_oscillator', # Awesome Oscillator

            # Trend indicators (7)
            'adx',                # Average Directional Index
            'aroon_up',           # Aroon Up
            'aroon_down',         # Aroon Down
            'ema_12',             # EMA(12)
            'ema_26',             # EMA(26)
            'sma_50',             # SMA(50)
            'sma_200',            # SMA(200)

            # Other indicators (5)
            'atr',                # Average True Range
            'keltner_upper',      # Keltner Channel Upper
            'keltner_lower',      # Keltner Channel Lower
            'sar',                # Parabolic SAR
            'supertrend',         # SuperTrend
        ]

        assert len(features) == 35, f"Indicator features: expected 35, got {len(features)}"
        return features

    def get_all_feature_columns(self) -> List[str]:
        """
        Get list of all 110 feature columns

        Returns:
            List of all feature column names in order
        """
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
        return [self.return_column]

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

    def get_feature_groups(self) -> dict:
        """
        Get feature groups for analysis

        Returns:
            Dict mapping group name to feature list
        """
        return {
            'orderbook': self.orderbook_features,
            'candle': self.candle_features,
            'indicator': self.indicator_features,
        }

    def summary(self) -> str:
        """
        Get summary of schema

        Returns:
            Formatted summary string
        """
        lines = [
            "Feature Store Schema Summary",
            "=" * 60,
            f"Total features: {len(self.get_all_feature_columns())}",
            f"  • OrderBook: {len(self.orderbook_features)}",
            f"  • Candle: {len(self.candle_features)}",
            f"  • Indicator: {len(self.indicator_features)}",
            "",
            f"Meta columns: {self.timestamp_column}, {self.symbol_column}",
            f"Label column: {self.label_column}",
            f"Return column: {self.return_column}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ===================================================================
# Global Default Schema
# ===================================================================

DEFAULT_SCHEMA = FeatureStoreSchema()

logger.info("✓ FeatureStoreSchema initialized with 110 features")
logger.debug(DEFAULT_SCHEMA.summary())
