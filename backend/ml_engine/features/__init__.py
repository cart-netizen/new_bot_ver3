"""
Feature extraction модуль.

Извлечение признаков из различных источников данных:
- OrderBook features (стакан ордеров)
- Candle features (свечные данные)
- Indicator features (технические индикаторы)
- Feature Pipeline (оркестрация)
"""

from .orderbook_feature_extractor import (
    OrderBookFeatureExtractor,
    OrderBookFeatures
)
from .candle_feature_extractor import (
    CandleFeatureExtractor,
    CandleFeatures,
    Candle
)
from .indicator_feature_extractor import (
    IndicatorFeatureExtractor,
    IndicatorFeatures
)
from .feature_pipeline import (
    FeaturePipeline,
    FeatureVector,
    MultiSymbolFeaturePipeline
)

__all__ = [
    # OrderBook
    "OrderBookFeatureExtractor",
    "OrderBookFeatures",
    # Candle
    "CandleFeatureExtractor",
    "CandleFeatures",
    "Candle",
    # Indicator
    "IndicatorFeatureExtractor",
    "IndicatorFeatures",
    # Pipeline
    "FeaturePipeline",
    "FeatureVector",
    "MultiSymbolFeaturePipeline"
]