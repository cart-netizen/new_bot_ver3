"""
Layering Data Collector for ML Training.

Professional data collection system for training adaptive ML models
to improve layering detection accuracy.

Features:
1. Automatic data collection in ONLY_TRAINING and full mode
2. Parquet storage for efficient ML pipelines
3. Feature extraction from layering patterns
4. Market context capture (regime, volatility, liquidity)
5. Label management (true positives, false positives)
6. Data quality validation
7. Train/validation split preparation

Path: backend/ml_engine/detection/layering_data_collector.py
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import json

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LayeringDataPoint:
  """Single training data point for ML model."""
  # Unique ID
  data_id: str
  timestamp: int
  symbol: str

  # ===== FEATURES (X) =====
  # Pattern features
  total_volume_btc: float
  total_volume_usdt: float
  placement_duration: float
  cancellation_rate: float
  spoofing_execution_ratio: Optional[float]
  layer_count: int
  total_orders: int
  avg_order_size: float
  price_spread_bps: float

  # Market context features
  market_regime: str  # "bullish", "bearish", "sideways", "unknown"
  volatility_24h: float  # 24h price volatility (%)
  volume_24h: float  # 24h trading volume
  liquidity_score: float  # Market depth score
  spread_bps: float  # Current bid-ask spread
  hour_utc: int  # Hour of day (0-23)
  day_of_week: int  # Day of week (0-6)

  # Price impact
  price_change_bps: Optional[float]
  expected_impact_bps: Optional[float]
  impact_ratio: Optional[float]

  # Execution metrics (if available)
  execution_volume: Optional[float]
  execution_trade_count: Optional[int]
  aggressive_ratio: Optional[float]

  # ===== LABELS (y) =====
  # Detector confidence (raw)
  detector_confidence: float

  # Human/automatic label (for training)
  is_true_layering: Optional[bool]  # None = unlabeled, True = confirmed, False = false positive
  label_source: str  # "automatic", "manual", "price_action", "unknown"
  label_confidence: float  # Confidence in the label (0.0-1.0)

  # Validation outcome (if available)
  price_moved_as_expected: Optional[bool]
  manipulation_successful: Optional[bool]

  # Component scores (for analysis)
  volume_score: float
  timing_score: float
  cancellation_score: float
  execution_correlation_score: float
  price_impact_score: float

  # Metadata
  notes: str = ""


class LayeringDataCollector:
  """
  Professional data collector for ML training.

  Collects data in both ONLY_TRAINING and full trading mode.
  Stores data in Parquet format for efficient ML pipelines.
  """

  def __init__(
      self,
      data_dir: str = "data/ml_training",
      enabled: bool = True,
      auto_save_interval: int = 100
  ):
    """
    Initialize data collector.

    Args:
        data_dir: Directory for storing training data
        enabled: Whether data collection is enabled
        auto_save_interval: Save to disk every N samples
    """
    self.data_dir = Path(data_dir)
    self.enabled = enabled
    self.auto_save_interval = auto_save_interval

    # In-memory buffer
    self.data_buffer: List[LayeringDataPoint] = []

    # Statistics
    self.total_collected = 0
    self.total_labeled = 0
    self.total_saved = 0

    if self.enabled:
      # Ensure directory exists
      self.data_dir.mkdir(parents=True, exist_ok=True)

      # Load existing data count
      self._load_statistics()

      logger.info(
        f"✅ LayeringDataCollector initialized: {self.data_dir}"
      )
      logger.info(
        f"   Total samples collected: {self.total_collected}"
      )

  def collect(
      self,
      pattern_data: Dict,
      market_context: Dict,
      component_scores: Dict,
      label: Optional[bool] = None,
      label_source: str = "automatic",
      label_confidence: float = 1.0
  ) -> str:
    """
    Collect a training data point.

    Args:
        pattern_data: Data from LayeringPattern
        market_context: Market regime, volatility, etc.
        component_scores: Individual confidence scores
        label: True positive / False positive (None = unlabeled)
        label_source: Source of label
        label_confidence: Confidence in label

    Returns:
        data_id: ID of collected sample
    """
    if not self.enabled:
      return ""

    # Generate unique ID
    timestamp = pattern_data.get('timestamp', int(datetime.now().timestamp() * 1000))
    symbol = pattern_data.get('symbol', 'UNKNOWN')
    data_id = f"{symbol}_{timestamp}_{self.total_collected}"

    # Extract features
    data_point = LayeringDataPoint(
      data_id=data_id,
      timestamp=timestamp,
      symbol=symbol,

      # Pattern features
      total_volume_btc=pattern_data.get('total_spoofing_volume', 0.0),
      total_volume_usdt=pattern_data.get('total_volume_usdt', 0.0),
      placement_duration=pattern_data.get('placement_duration', 0.0),
      cancellation_rate=pattern_data.get('cancellation_rate', 0.0),
      spoofing_execution_ratio=pattern_data.get('spoofing_execution_ratio'),
      layer_count=pattern_data.get('layer_count', 0),
      total_orders=pattern_data.get('total_orders', 0),
      avg_order_size=pattern_data.get('avg_order_size', 0.0),
      price_spread_bps=pattern_data.get('price_spread_bps', 0.0),

      # Market context
      market_regime=market_context.get('market_regime', 'unknown'),
      volatility_24h=market_context.get('volatility_24h', 0.0),
      volume_24h=market_context.get('volume_24h', 0.0),
      liquidity_score=market_context.get('liquidity_score', 0.0),
      spread_bps=market_context.get('spread_bps', 0.0),
      hour_utc=datetime.utcfromtimestamp(timestamp / 1000).hour,
      day_of_week=datetime.utcfromtimestamp(timestamp / 1000).weekday(),

      # Price impact
      price_change_bps=pattern_data.get('price_change_bps'),
      expected_impact_bps=pattern_data.get('expected_impact_bps'),
      impact_ratio=pattern_data.get('impact_ratio'),

      # Execution metrics
      execution_volume=pattern_data.get('execution_volume'),
      execution_trade_count=pattern_data.get('execution_trade_count'),
      aggressive_ratio=pattern_data.get('aggressive_ratio'),

      # Labels
      detector_confidence=pattern_data.get('confidence', 0.0),
      is_true_layering=label,
      label_source=label_source,
      label_confidence=label_confidence,
      price_moved_as_expected=pattern_data.get('price_moved_as_expected'),
      manipulation_successful=pattern_data.get('manipulation_successful'),

      # Component scores
      volume_score=component_scores.get('volume', 0.0),
      timing_score=component_scores.get('timing', 0.0),
      cancellation_score=component_scores.get('cancellation', 0.0),
      execution_correlation_score=component_scores.get('execution_correlation', 0.0),
      price_impact_score=component_scores.get('price_impact', 0.0),

      notes=""
    )

    # Add to buffer
    self.data_buffer.append(data_point)
    self.total_collected += 1

    if label is not None:
      self.total_labeled += 1

    # Auto-save if buffer full
    if len(self.data_buffer) >= self.auto_save_interval:
      self.save_to_disk()

    logger.debug(
      f"📊 Data collected: {data_id}, "
      f"confidence={data_point.detector_confidence:.2f}, "
      f"labeled={label is not None}"
    )

    return data_id

  def update_label(
      self,
      data_id: str,
      label: bool,
      label_source: str = "manual",
      label_confidence: float = 1.0,
      notes: str = ""
  ):
    """
    Update label for existing data point.

    Useful for manual labeling or post-validation.

    Args:
        data_id: ID of data point
        label: True/False
        label_source: Source of label update
        label_confidence: Confidence in new label
        notes: Additional notes
    """
    if not self.enabled:
      return

    # Check buffer first
    for i, data_point in enumerate(self.data_buffer):
      if data_point.data_id == data_id:
        self.data_buffer[i].is_true_layering = label
        self.data_buffer[i].label_source = label_source
        self.data_buffer[i].label_confidence = label_confidence
        if notes:
          self.data_buffer[i].notes = notes

        if data_point.is_true_layering is None:
          self.total_labeled += 1

        logger.info(f"🏷️  Label updated: {data_id}, label={label}")
        return

    # Check disk files (more expensive)
    self._update_label_in_files(data_id, label, label_source, label_confidence, notes)

  def save_to_disk(self):
    """Save buffer to Parquet file."""
    if not self.enabled or not self.data_buffer:
      return

    # Convert to DataFrame
    df = pd.DataFrame([asdict(dp) for dp in self.data_buffer])

    # File naming: layering_data_YYYYMMDD_HHMMSS.parquet
    filename = f"layering_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    filepath = self.data_dir / filename

    # Save to Parquet
    df.to_parquet(filepath, engine='pyarrow', compression='snappy')

    self.total_saved += len(self.data_buffer)

    logger.info(
      f"💾 Saved {len(self.data_buffer)} samples to {filename}"
    )

    # Clear buffer
    self.data_buffer.clear()

    # Update statistics
    self._save_statistics()

  def load_all_data(self) -> pd.DataFrame:
    """
    Load all collected data from disk.

    Returns:
        DataFrame with all training data
    """
    if not self.enabled:
      return pd.DataFrame()

    # Find all parquet files
    parquet_files = list(self.data_dir.glob("layering_data_*.parquet"))

    if not parquet_files:
      logger.warning("No training data files found")
      return pd.DataFrame()

    # Load and concat
    dfs = []
    for filepath in parquet_files:
      try:
        df = pd.read_parquet(filepath)
        # Only add non-empty DataFrames with actual data to avoid FutureWarning
        # Filter out both empty DataFrames and those with all-NA columns
        if not df.empty and not df.isna().all().all():
          dfs.append(df)
      except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")

    if not dfs:
      return pd.DataFrame()

    # Suppress FutureWarning for concat operation
    import warnings
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
      combined_df = pd.concat(dfs, ignore_index=True)

    logger.info(
      f"📚 Loaded {len(combined_df)} samples from {len(dfs)} files"
    )

    return combined_df

  def get_labeled_data(self) -> pd.DataFrame:
    """
    Get only labeled data (for supervised learning).

    Returns:
        DataFrame with labeled samples only
    """
    df = self.load_all_data()

    if df.empty:
      return df

    # Filter labeled samples
    labeled_df = df[df['is_true_layering'].notna()].copy()

    logger.info(
      f"🏷️  Labeled data: {len(labeled_df)} / {len(df)} "
      f"({len(labeled_df) / len(df) * 100:.1f}%)"
    )

    return labeled_df

  def get_training_split(
      self,
      test_size: float = 0.2,
      random_state: int = 42
  ) -> Dict[str, pd.DataFrame]:
    """
    Get train/test split for ML training.

    Args:
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        Dict with 'train' and 'test' DataFrames
    """
    labeled_df = self.get_labeled_data()

    if len(labeled_df) < 10:
      logger.warning(f"Insufficient labeled data: {len(labeled_df)} samples")
      return {'train': pd.DataFrame(), 'test': pd.DataFrame()}

    # Split
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
      labeled_df,
      test_size=test_size,
      random_state=random_state,
      stratify=labeled_df['is_true_layering']  # Balanced split
    )

    logger.info(
      f"📊 Training split: train={len(train_df)}, test={len(test_df)}"
    )

    return {'train': train_df, 'test': test_df}

  def _update_label_in_files(
      self,
      data_id: str,
      label: bool,
      label_source: str,
      label_confidence: float,
      notes: str
  ):
    """Update label in Parquet files on disk."""
    parquet_files = list(self.data_dir.glob("layering_data_*.parquet"))

    for filepath in parquet_files:
      try:
        df = pd.read_parquet(filepath)

        if data_id in df['data_id'].values:
          # Update label
          df.loc[df['data_id'] == data_id, 'is_true_layering'] = label
          df.loc[df['data_id'] == data_id, 'label_source'] = label_source
          df.loc[df['data_id'] == data_id, 'label_confidence'] = label_confidence
          if notes:
            df.loc[df['data_id'] == data_id, 'notes'] = notes

          # Save back
          df.to_parquet(filepath, engine='pyarrow', compression='snappy')

          logger.info(f"🏷️  Label updated in {filepath.name}: {data_id}")
          return

      except Exception as e:
        logger.error(f"Error updating {filepath}: {e}")

  def _save_statistics(self):
    """Save collection statistics."""
    stats_file = self.data_dir / "statistics.json"

    stats = {
      'total_collected': self.total_collected,
      'total_labeled': self.total_labeled,
      'total_saved': self.total_saved,
      'last_updated': datetime.now().isoformat()
    }

    with open(stats_file, 'w') as f:
      json.dump(stats, f, indent=2)

  def _load_statistics(self):
    """Load collection statistics."""
    stats_file = self.data_dir / "statistics.json"

    if not stats_file.exists():
      return

    try:
      with open(stats_file, 'r') as f:
        stats = json.load(f)

      self.total_collected = stats.get('total_collected', 0)
      self.total_labeled = stats.get('total_labeled', 0)
      self.total_saved = stats.get('total_saved', 0)

    except Exception as e:
      logger.error(f"Error loading statistics: {e}")

  def get_statistics(self) -> Dict:
    """Get collection statistics."""
    all_df = self.load_all_data()
    labeled_df = all_df[all_df['is_true_layering'].notna()] if not all_df.empty else pd.DataFrame()

    if not labeled_df.empty:
      true_positive_count = int(labeled_df['is_true_layering'].sum())
      false_positive_count = len(labeled_df) - true_positive_count
      labeling_rate = len(labeled_df) / len(all_df) if len(all_df) > 0 else 0.0
    else:
      true_positive_count = 0
      false_positive_count = 0
      labeling_rate = 0.0

    return {
      'enabled': self.enabled,
      'total_collected': self.total_collected,
      'total_labeled': self.total_labeled,
      'total_saved': self.total_saved,
      'buffer_size': len(self.data_buffer),
      'files_on_disk': len(list(self.data_dir.glob("layering_data_*.parquet"))),
      'total_on_disk': len(all_df) if not all_df.empty else 0,
      'labeled_on_disk': len(labeled_df) if not labeled_df.empty else 0,
      'labeling_rate': labeling_rate,
      'true_positives': true_positive_count,
      'false_positives': false_positive_count
    }

  def export_for_training(self, output_path: str):
    """
    Export labeled data in format ready for ML training.

    Args:
        output_path: Path to output Parquet file
    """
    labeled_df = self.get_labeled_data()

    if labeled_df.empty:
      logger.warning("No labeled data to export")
      return

    # Save
    labeled_df.to_parquet(output_path, engine='pyarrow', compression='snappy')

    logger.info(
      f"📦 Exported {len(labeled_df)} labeled samples to {output_path}"
    )


# Example usage
if __name__ == "__main__":
  collector = LayeringDataCollector()

  # Simulate data collection
  pattern_data = {
    'timestamp': int(datetime.now().timestamp() * 1000),
    'symbol': 'BTCUSDT',
    'total_spoofing_volume': 5.2,
    'total_volume_usdt': 260000,
    'placement_duration': 12.5,
    'cancellation_rate': 0.73,
    'spoofing_execution_ratio': 10.5,
    'layer_count': 3,
    'total_orders': 15,
    'confidence': 0.85
  }

  market_context = {
    'market_regime': 'bullish',
    'volatility_24h': 2.5,
    'volume_24h': 15000000,
    'liquidity_score': 0.8,
    'spread_bps': 0.5
  }

  component_scores = {
    'volume': 0.20,
    'timing': 0.15,
    'cancellation': 0.25,
    'execution_correlation': 0.15,
    'price_impact': 0.10
  }

  data_id = collector.collect(
    pattern_data,
    market_context,
    component_scores,
    label=True,
    label_source="automatic"
  )

  print(f"Collected: {data_id}")

  # Save
  collector.save_to_disk()

  # Statistics
  stats = collector.get_statistics()
  print(f"Statistics: {stats}")
