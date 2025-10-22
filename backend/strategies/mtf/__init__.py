"""
Multi-Timeframe Analysis Module.

Компоненты:
- TimeframeCoordinator: управление свечами для множественных TF
- TimeframeAnalyzer: независимый анализ каждого TF
- TimeframeAligner: проверка согласованности между TF
- TimeframeSignalSynthesizer: синтез финального MTF сигнала
- MultiTimeframeManager: главный оркестратор всех MTF компонентов

Путь: backend/strategies/mtf/__init__.py
"""

from .timeframe_coordinator import (
  TimeframeCoordinator,
  Timeframe,
  TimeframeConfig,
  MultiTimeframeConfig
)

from .timeframe_analyzer import (
  TimeframeAnalyzer,
  TimeframeAnalysisResult,
  TimeframeIndicators,
  TimeframeRegimeInfo,
  MarketRegime,
  VolatilityRegime
)

from .timeframe_aligner import (
  TimeframeAligner,
  TimeframeAlignment,
  AlignmentConfig,
  AlignmentType,
  DivergenceType,
  ConfluenceZone
)

from .timeframe_signal_synthesizer import (
  TimeframeSignalSynthesizer,
  MultiTimeframeSignal,
  SynthesizerConfig,
  SynthesisMode
)

from .multi_timeframe_manager import (
  MultiTimeframeManager,
  MTFManagerConfig
)

__all__ = [
  # Coordinator
  'TimeframeCoordinator',
  'Timeframe',
  'TimeframeConfig',
  'MultiTimeframeConfig',

  # Analyzer
  'TimeframeAnalyzer',
  'TimeframeAnalysisResult',
  'TimeframeIndicators',
  'TimeframeRegimeInfo',
  'MarketRegime',
  'VolatilityRegime',

  # Aligner
  'TimeframeAligner',
  'TimeframeAlignment',
  'AlignmentConfig',
  'AlignmentType',
  'DivergenceType',
  'ConfluenceZone',

  # Synthesizer
  'TimeframeSignalSynthesizer',
  'MultiTimeframeSignal',
  'SynthesizerConfig',
  'SynthesisMode',

  # Manager
  'MultiTimeframeManager',
  'MTFManagerConfig',
]